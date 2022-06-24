import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname(os.path.abspath(__file__))
dataset_config = BaseDatasetConfig(
    name="sleeping_ce", meta_file_train="metadata.csv", path=os.path.join(output_path, "../filelists")
)
audio_config = BaseAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    preemphasis=0.98,
    ref_level_db=20,
    log_func="np.log",
    do_trim_silence=True,
    trim_db=60,
    mel_fmin=0,
    mel_fmax=8000,
    spec_gain=20,
    signal_norm=False,
    do_amp_to_db_linear=False,
)

config = VitsConfig(
    audio=audio_config,
    run_name="vits_sleeping_ce",
    batch_size=16,
    eval_batch_size=16,
    batch_group_size=5,
    num_loader_workers=0,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="korean_cleaners",
    use_phonemes=False,
    phoneme_language="ko",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    save_step=1000,
    save_best_after=1000,
    print_step=25,
    print_eval=True,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters=" .!?\u1100\u1101\u1102\u1103\u1104\u1105\u1106\u1107\u1108\u1109\u110a\u110b\u110c\u110d\u110e\u110f\u1110\u1111\u1112\u1161\u1162\u1163\u1164\u1165\u1166\u1167\u1168\u1169\u116a\u116b\u116c\u116d\u116e\u116f\u1170\u1171\u1172\u1173\u1174\u1175\u11a8\u11a9\u11aa\u11ab\u11ac\u11ad\u11ae\u11af\u11b0\u11b1\u11b2\u11b3\u11b4\u11b5\u11b6\u11b7\u11b8\u11b9\u11ba\u11bb\u11bc\u11bd\u11be\u11bf\u11c0\u11c1\u11c2",
        punctuations=" .!?",
        phonemes="\u1100\u1101\u1102\u1103\u1104\u1105\u1106\u1107\u1108\u1109\u110a\u110b\u110c\u110d\u110e\u110f\u1110\u1111\u1112\u1161\u1162\u1163\u1164\u1165\u1166\u1167\u1168\u1169\u116a\u116b\u116c\u116d\u116e\u116f\u1170\u1171\u1172\u1173\u1174\u1175\u11a8\u11a9\u11aa\u11ab\u11ac\u11ad\u11ae\u11af\u11b0\u11b1\u11b2\u11b3\u11b4\u11b5\u11b6\u11b7\u11b8\u11b9\u11ba\u11bb\u11bc\u11bd\u11be\u11bf\u11c0\u11c1\u11c2",
    ),
    test_sentences=[
        [
            "아래 문장들은 모델 학습을 위해 사용하지 않은 문장들입니다.",
            "sleeping_ce",
            None,
            "ko",
        ],
        [
            "서울특별시 특허허가과 허가과장 허과장.",
            "sleeping_ce",
            None,
            "ko",
        ],
        [
            "경찰청 철창살은 외철창살이고 검찰청 철창살은 쌍철창살이다.",
            "sleeping_ce",
            None,
            "ko",
        ],
        [
            "지향을 지양으로 오기하는 일을 지양하는 언어 습관을 지향해야 한다.",
            "sleeping_ce",
            None,
            "ko",
        ],
        [
            "그러니까 외계인이 우리 생각을 읽고 우리 생각을 우리가 다시 생각토록 해서 그 생각이 마치 우리가 생각한 것인 것처럼 속였다는 거냐?",
            "sleeping_ce",
            None,
            "ko",
        ],
    ],
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init model
model = Vits(config, ap, tokenizer, speaker_manager=None)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
