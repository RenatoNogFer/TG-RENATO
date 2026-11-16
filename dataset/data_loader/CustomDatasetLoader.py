"""Custom Dataset dataloader for TCC automotive rPPG dataset.

Expected per-subject/session structure:

DATASET_TCC/
  subject_or_session_id/
    primary_video.mp4
    secondary_video.mp4                  # optional, ignored by this loader
    hr_sync.csv                          # columns: heart_rate_bpm, relative_video_time
    primary_video_auto_anotacoes.txt     # optional, used later by metrics.py
    sync.txt                             # optional, ignored by this loader

This loader preprocesses primary_video.mp4 and stores a preprocess_metadata.csv file
inside each subject/session folder to make chunks traceable back to the original video.
"""

import glob
import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch

from dataset.data_loader.BaseLoader import BaseLoader

try:
    from dataset.data_loader.face_detector.YOLO5Face import YOLO5Face
except ImportError:
    try:
        from dataset.face_detector.YOLO5Face import YOLO5Face
    except ImportError:
        sys.path.append(os.getcwd())
        try:
            from YOLO5Face import YOLO5Face
        except ImportError:
            YOLO5Face = None


class CustomDatasetLoader(BaseLoader):
    def __init__(self, name, data_path, config_data, device=None):
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Return one entry per subject/session folder."""
        all_items = glob.glob(data_path + os.sep + "*")
        data_dirs = [d for d in all_items if os.path.isdir(d)]
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")

        dirs = [
            {
                "index": str(i),
                "path": data_dir,
                "name": os.path.basename(data_dir),
                "subject_id": os.path.basename(data_dir),
            }
            for i, data_dir in enumerate(sorted(data_dirs))
        ]
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        if begin == 0 and end == 1:
            return data_dirs
        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        return [data_dirs[i] for i in choose_range]

    def _generate_bpm_label_from_csv(self, csv_path, total_frames, fs):
        """Generate a per-frame BPM label from hr_sync.csv.

        Expected columns:
            - heart_rate_bpm
            - relative_video_time

        timestamp_brt, when present, is intentionally ignored.
        """
        try:
            df = pd.read_csv(csv_path)

            required_cols = {"heart_rate_bpm", "relative_video_time"}
            missing_cols = required_cols.difference(df.columns)
            if missing_cols:
                raise ValueError(
                    f"Missing required columns {sorted(missing_cols)}. "
                    f"Expected columns: {sorted(required_cols)}"
                )

            bpm_vals = pd.to_numeric(df["heart_rate_bpm"], errors="coerce").to_numpy()
            rel_time = pd.to_numeric(df["relative_video_time"], errors="coerce").to_numpy()

            valid = ~np.isnan(bpm_vals) & ~np.isnan(rel_time)
            bpm_vals = bpm_vals[valid]
            rel_time = rel_time[valid]

            if len(bpm_vals) == 0:
                return None

            order = np.argsort(rel_time)
            rel_time = rel_time[order]
            bpm_vals = bpm_vals[order]

            # Remove duplicate timestamps, keeping the last BPM for each timestamp.
            unique_times, unique_indices = np.unique(rel_time, return_index=True)
            if len(unique_times) != len(rel_time):
                tmp = pd.DataFrame({"t": rel_time, "bpm": bpm_vals})
                tmp = tmp.groupby("t", as_index=False).last()
                rel_time = tmp["t"].to_numpy()
                bpm_vals = tmp["bpm"].to_numpy()

            frame_times = np.arange(total_frames) / fs
            bpm_per_frame = np.interp(frame_times, rel_time, bpm_vals)
            return bpm_per_frame.astype(np.float32)

        except Exception as e:
            print(f"[AVISO] Falha ao gerar label BPM a partir de {csv_path}: {e}")
            return None

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        self.detector = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dir_info = data_dirs[i]
        saved_base_filename = dir_info["index"]
        subject_id = dir_info.get("subject_id", dir_info["name"])

        print(f"[PRE-PROC] Processando: {dir_info['name']}...")

        # Current TCC dataset structure.
        video_path = os.path.join(dir_info["path"], "primary_video.mp4")
        csv_path = os.path.join(dir_info["path"], "hr_sync.csv")
        failure_annotation_path = os.path.join(
            dir_info["path"], "primary_video_auto_anotacoes.txt"
        )
        metadata_path = os.path.join(dir_info["path"], "preprocess_metadata.csv")

        if not os.path.exists(video_path):
            print(f"[ERRO CRÍTICO] primary_video.mp4 não encontrado em: {dir_info['path']}")
            return

        fs = self.config_data.FS

        cap_temp = cv2.VideoCapture(video_path)
        if not cap_temp.isOpened():
            print(f"[ERRO CRÍTICO] Não foi possível abrir o vídeo: {video_path}")
            return

        total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        source_video_fps = float(cap_temp.get(cv2.CAP_PROP_FPS) or 0.0)
        cap_temp.release()

        full_gt = None
        self.using_bpm_labels = False

        use_pseudo_ppg = getattr(config_preprocess, "USE_PSUEDO_PPG_LABEL", False)
        if not use_pseudo_ppg:
            if os.path.exists(csv_path):
                full_gt = self._generate_bpm_label_from_csv(csv_path, total_frames, fs)
                if full_gt is not None:
                    self.using_bpm_labels = True
            else:
                print(f"[AVISO] Arquivo de HR não encontrado: {csv_path}")

        chunk_len = config_preprocess.CHUNK_LENGTH
        if chunk_len <= 0:
            chunk_len = 180

        cap = cv2.VideoCapture(video_path)
        frames_buffer = []
        frame_info_buffer = []
        batch_counter = 0
        global_frame_counter = 0
        all_input_names = []
        metadata_records = []

        yolo_detector = None
        try:
            if YOLO5Face is not None:
                device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
                with torch.no_grad():
                    yolo_detector = YOLO5Face(backend="Y5F", device=device_str)
        except Exception:
            yolo_detector = None

        frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        haar_frontal = cv2.CascadeClassifier(frontal_path)
        haar_profile = cv2.CascadeClassifier(profile_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None or frame.size == 0:
                continue

            original_h, original_w = frame.shape[:2]
            resize_scale = 1.0

            if original_w > 1920:
                resize_scale = 1920 / original_w
                frame = cv2.resize(frame, (1920, int(original_h * resize_scale)))

            processed_h, processed_w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames_buffer.append(frame_rgb)
            frame_info_buffer.append(
                {
                    "original_frame_width": original_w,
                    "original_frame_height": original_h,
                    "processed_frame_width": processed_w,
                    "processed_frame_height": processed_h,
                    "resize_scale": resize_scale,
                }
            )
            global_frame_counter += 1

            if len(frames_buffer) == chunk_len:
                current_crop_box = None
                detector_name = "none"
                face_detected = False
                fallback_used = False

                if yolo_detector is not None:
                    with torch.no_grad():
                        current_crop_box = self._detect_yolo_native(frames_buffer[0], yolo_detector)
                    if current_crop_box is not None:
                        detector_name = "YOLO5Face"
                        face_detected = True

                if current_crop_box is None:
                    current_crop_box = self._detect_haar_robust(
                        frames_buffer[0], haar_frontal, haar_profile
                    )
                    if current_crop_box is not None:
                        detector_name = "HaarCascade"
                        face_detected = True

                if current_crop_box is None:
                    h_img, w_img = frames_buffer[0].shape[:2]
                    current_crop_box = (w_img // 4, h_img // 4, w_img // 2, h_img // 2)
                    detector_name = "center_fallback"
                    face_detected = False
                    fallback_used = True

                first_frame_info = frame_info_buffer[0]

                self._process_and_save_chunk(
                    frames_list=frames_buffer,
                    crop_box=current_crop_box,
                    full_gt=full_gt,
                    base_name=saved_base_filename,
                    batch_idx=batch_counter,
                    current_frame_pos=global_frame_counter,
                    chunk_len=chunk_len,
                    config=config_preprocess,
                    name_list=all_input_names,
                    fs=fs,
                    metadata_records=metadata_records,
                    dir_info=dir_info,
                    subject_id=subject_id,
                    video_path=video_path,
                    csv_path=csv_path,
                    failure_annotation_path=failure_annotation_path,
                    total_frames=total_frames,
                    source_video_fps=source_video_fps,
                    frame_info=first_frame_info,
                    detector_name=detector_name,
                    face_detected=face_detected,
                    fallback_used=fallback_used,
                )

                frames_buffer = []
                frame_info_buffer = []
                batch_counter += 1

        cap.release()
        file_list_dict[i] = all_input_names

        # Save metadata even when no chunks are processed, to make failures visible.
        try:
            pd.DataFrame(metadata_records).to_csv(metadata_path, index=False)
        except Exception as e:
            print(f"[AVISO] Não foi possível salvar metadata em {metadata_path}: {e}")

        del yolo_detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _detect_yolo_native(self, frame, yolo_detector):
        try:
            res = yolo_detector.detect_face(frame)
            if res is not None:
                x1, y1, x2, y2 = res
                w = x2 - x1
                h = y2 - y1
                if w < 20 or h < 20:
                    return None
                return (int(x1), int(y1), int(w), int(h))
        except Exception:
            return None
        return None

    def _detect_haar_robust(self, frame, haar_frontal, haar_profile):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        h_img, w_img = frame.shape[:2]

        faces = haar_frontal.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            return self._get_largest_face(faces, w_img, h_img)

        faces = haar_profile.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            return self._get_largest_face(faces, w_img, h_img)

        gray_flipped = cv2.flip(gray, 1)
        faces_flipped = haar_profile.detectMultiScale(gray_flipped, 1.3, 5)
        if len(faces_flipped) > 0:
            faces = []
            for (x, y, w, h) in faces_flipped:
                faces.append((w_img - x - w, y, w, h))
            return self._get_largest_face(faces, w_img, h_img)

        return None

    def _get_largest_face(self, faces, w_img, h_img):
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        x = max(0, x - int(w * 0.05))
        y = max(0, y - int(h * 0.05))
        w = min(w_img - x, int(w * 1.1))
        h = min(h_img - y, int(h * 1.1))
        return (x, y, w, h)

    def _process_and_save_chunk(
        self,
        frames_list,
        crop_box,
        full_gt,
        base_name,
        batch_idx,
        current_frame_pos,
        chunk_len,
        config,
        name_list,
        fs,
        metadata_records,
        dir_info,
        subject_id,
        video_path,
        csv_path,
        failure_annotation_path,
        total_frames,
        source_video_fps,
        frame_info,
        detector_name,
        face_detected,
        fallback_used,
    ):
        x, y, w, h = crop_box
        if w < 10 or h < 10:
            w, h = 64, 64

        cropped_frames = []
        target_h, target_w = config.RESIZE.H, config.RESIZE.W

        for f in frames_list:
            crop = f[y : y + h, x : x + w]
            if crop.size == 0:
                crop = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
            cropped_frames.append(resized)

        frames_arr = np.asarray(cropped_frames)

        start_idx = current_frame_pos - chunk_len
        end_idx = current_frame_pos

        use_pseudo_ppg = getattr(config, "USE_PSUEDO_PPG_LABEL", False)
        if use_pseudo_ppg:
            gt_chunk = self.generate_pos_psuedo_labels(frames_arr, fs=fs)
            label_source = "pseudo_ppg"
        else:
            if full_gt is not None:
                if start_idx < len(full_gt):
                    slice_end = min(end_idx, len(full_gt))
                    gt_chunk = full_gt[start_idx:slice_end]
                    if len(gt_chunk) < chunk_len:
                        gt_chunk = np.pad(gt_chunk, (0, chunk_len - len(gt_chunk)), "edge")
                else:
                    gt_chunk = np.full(chunk_len, full_gt[-1] if len(full_gt) > 0 else 0)
                label_source = "hr_sync_csv"
            else:
                duration_sec = np.arange(start_idx, end_idx) / fs
                gt_chunk = np.sin(2 * np.pi * 1.2 * duration_sec)
                label_source = "synthetic_fallback"

        if len(gt_chunk) != len(frames_arr):
            gt_chunk = BaseLoader.resample_ppg(gt_chunk, len(frames_arr))

        gt_chunk = gt_chunk.astype(np.float32)

        config_clone = config.clone()
        config_clone.defrost()
        config_clone.DO_CHUNK = False
        config_clone.CROP_FACE.DO_CROP_FACE = False

        # If the reference comes from BPM labels, prevent DiffNormalized processing
        # from being applied to labels as if they were BVP waveforms.
        if hasattr(self, "using_bpm_labels") and self.using_bpm_labels:
            config_clone.LABEL_TYPE = "Raw"

        config_clone.freeze()

        frames_clips, bvps_clips = self.preprocess(frames_arr, gt_chunk, config_clone)
        unique_filename = f"{base_name}_{batch_idx}"
        input_list, label_list = self.save_multi_process(
            frames_clips, bvps_clips, unique_filename
        )
        name_list.extend(input_list)

        metadata_records.append(
            {
                "video_idx": base_name,
                "subject_id": subject_id,
                "video_name": dir_info["name"],
                "video_dir": dir_info["path"],
                "video_path": video_path,
                "hr_csv_path": csv_path if os.path.exists(csv_path) else "",
                "failure_annotation_path": failure_annotation_path
                if os.path.exists(failure_annotation_path)
                else "",
                "chunk_idx": batch_idx,
                "saved_base_filename": unique_filename,
                "saved_input_files": "|".join(input_list) if input_list else "",
                "saved_label_files": "|".join(label_list) if label_list else "",
                "start_frame": start_idx,
                "end_frame_exclusive": end_idx,
                "start_time_sec": start_idx / fs,
                "end_time_sec": end_idx / fs,
                "chunk_length_frames": chunk_len,
                "config_fs": fs,
                "video_total_frames": total_frames,
                "source_video_fps": source_video_fps,
                "original_frame_width": frame_info.get("original_frame_width"),
                "original_frame_height": frame_info.get("original_frame_height"),
                "processed_frame_width": frame_info.get("processed_frame_width"),
                "processed_frame_height": frame_info.get("processed_frame_height"),
                "resize_scale": frame_info.get("resize_scale"),
                "crop_x": int(x),
                "crop_y": int(y),
                "crop_w": int(w),
                "crop_h": int(h),
                "detector": detector_name,
                "face_detected": bool(face_detected),
                "fallback_used": bool(fallback_used),
                "label_source": label_source,
                "preprocess_target_width": target_w,
                "preprocess_target_height": target_h,
            }
        )

    @staticmethod
    def read_wave(bvp_file):
        """Kept for compatibility with older experiments, but not used by the TCC structure."""
        try:
            with open(bvp_file, "r") as f:
                content = f.read().strip()
                if not content:
                    return np.array([0.0])
                if "\n" in content and len(content.split("\n")) > 1:
                    bvp = [float(x) for x in content.split("\n") if x.strip()]
                else:
                    bvp = [float(x) for x in content.split() if x.strip()]
            return np.asarray(bvp)
        except Exception:
            return np.array([0.0])
