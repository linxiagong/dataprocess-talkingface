import numpy as np
import torch
import torchaudio


class HubertFeatureExtractor:
    def __init__(self, ckpt: str = None) -> None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.device = device

        # https://github.com/bshall/hubert
        # Load checkpoint (either hubert_soft or hubert_discrete)
        model = torch.hub.load("bshall/hubert:main", "hubert_soft")
        self.model = model.to(self.device)
        self._model_sample_rate = 16000

    def extract_hubert_features(self, audio_file_path: str, target_len: int = None, target_fps: int = None):
        # load tensor from file
        waveform, sample_rate = torchaudio.load(audio_file_path)
        # Resample audio to the expected sampling rate
        if sample_rate != self._model_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self._model_sample_rate)

        # Extract speech units
        waveform = waveform.unsqueeze(0).to(self.device)
        features = self.model.units(waveform)  # [1, audio_units, feat_len]
        features = features.squeeze(0).cpu().numpy()

        # Align the audio units with the images number
        audio_fps = 50  # hubert extract one unit per 20ms
        if target_len and target_fps:
            features = self.interpolate_features(features=features,
                                                 src_fps=audio_fps,
                                                 target_len=target_len,
                                                 target_fps=target_fps)
        return self.postprocess(features)

    def interpolate_features(self, features: np.array, src_fps: int, target_len: int, target_fps: int) -> np.array:
        """
        Interpolate HuBert features.
        """
        src_len, feat_len = features.shape
        src_timestamps = np.arange(src_len) / float(src_fps)
        target_timestamps = np.arange(target_len) / float(target_fps)
        output_features = np.zeros((target_len, feat_len))
        for feature_idx in range(feat_len):
            output_features[:, feature_idx] = np.interp(x=target_timestamps,
                                                        xp=src_timestamps,
                                                        fp=features[:, feature_idx])
        return output_features

    def postprocess(self, features, win_size: int = 16):
        """
        Pack the features into a shape of [audio_units, win_size, feat_len],
        for the ease of later usage.
        """
        src_len, feat_len = features.shape

        zero_pad = np.zeros((int(win_size / 2), feat_len))
        features = np.concatenate((zero_pad, features, zero_pad), axis=0)
        windows = []
        for window_index in range(0, src_len):
            windows.append(features[window_index:window_index + win_size])
        windows = np.array(windows)
        return windows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_file', type=str, help="path to wav file")
    args = parser.parse_args()

    hubert_extractor = HubertFeatureExtractor()
    output_features = hubert_extractor.extract_hubert_features(args.wav_file)
