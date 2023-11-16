import numpy as np
import torch
import os

os.environ['TRANSFORMERS_CACHE'] = 'cache'
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoConfig, \
    AutoModelForSequenceClassification, AutoTokenizer
from pytube import YouTube


def all_model_info_roberta():
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    return MODEL, model, tokenizer, config


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def get_transcript(path):
    """
    Get audio transcript from audio file
    :param path:
    :type path:
    :return:
    :rtype:
    """

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "distil-whisper/distil-medium.en"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=15,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )
    result = pipe(path)
    print(result['text'])
    os.remove(path)
    return result


def download_yt_video(link):
    """
    Download YouTube video (link should be pre-validated)
    :param link:
    :type link:
    :return:
    :rtype:
    """
    try:
        selected_video = YouTube(link)
        if os.path.exists(f'audio/{selected_video.video_id}.mp4'):
            os.remove(f'audio/{selected_video.video_id}.mp4')
        audio_only = selected_video.streams.filter(only_audio=True).first()
        try:
            audio_only.download('audio/', filename=f'{selected_video.video_id}.mp4')
        except Exception as e:
            raise InterruptedError(e)
        return f'audio/{selected_video.video_id}.mp4'
    except Exception as e:
        raise InterruptedError(e)


def summery_video(text):
    """
    Summarize text from the video
    :param text:
    :type text:
    :return:
    :rtype:
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, min_length=100, do_sample=False)
    return summary


def get_sentiment(text):
    """
    Get the sentiment of the text
    :param text:
    :type text:
    :return:
    :rtype:
    """
    MODEL, model, tokenizer, config = all_model_info_roberta()

    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax_stable(scores)
    return scores  # bad, neutral, good


def softmax_stable(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

#
# if __name__ == '__main__':
#     try:
#         import time
#
#         t_ = time.time()
#         d = download_yt_video(link='https://www.youtube.com/watch?v=U-GC3Z2OKv4')
#         print(f"Time took to download: {time.time() - t_}")
#         t_ = time.time()
#         text = get_transcript(path=d)['text']
#         print(text)
#         print(f"Time took to transcribe: {time.time() - t_}")
#         t_ = time.time()
#         summ = summery_video(text=text)[0]['summary_text']
#         print(summ)
#         print(f"Time took to summarize: {time.time() - t_}")
#         t_ = time.time()
#         sent = get_sentiment(text=summ)
#         print(sent)
#         print(f"Time took to analyze: {time.time() - t_}")
#     except Exception as e:
#         print(e)
