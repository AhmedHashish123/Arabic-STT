from jiwer import wer

def calculate_wer(common_voice, model_1, model_2, model_3, model_4, model_5):
  total_wer_1 = 0
  total_wer_2 = 0
  total_wer_3 = 0
  total_wer_4 = 0
  total_wer_5 = 0
  counter = 0
  for i in range(len(common_voice)):
    reference = common_voice[i]['sentence']
    hypothesis_1 = model_1.transcribe(common_voice[i]['path'])
    hypothesis_2 = model_2.transcribe(common_voice[i]['path'])
    hypothesis_3 = model_3.transcribe(common_voice[i]['path'])
    hypothesis_4 = model_4.transcribe(common_voice[i]['path'])
    hypothesis_5 = model_5.transcribe(common_voice[i]['path'])
    wer_1 = wer(reference, hypothesis_1)
    wer_2 = wer(reference, hypothesis_2)
    wer_3 = wer(reference, hypothesis_3)
    wer_4 = wer(reference, hypothesis_4)
    wer_5 = wer(reference, hypothesis_5)
    total_wer_1 += wer_1
    total_wer_2 += wer_2
    total_wer_3 += wer_3
    total_wer_4 += wer_4
    total_wer_5 += wer_5
    counter += 1

  avg_wer_1 = total_wer_1 / len(common_voice)
  avg_wer_2 = total_wer_2 / len(common_voice)
  avg_wer_3 = total_wer_3 / len(common_voice)
  avg_wer_4 = total_wer_4 / len(common_voice)
  avg_wer_5 = total_wer_5 / len(common_voice)

  print(avg_wer_1)
  print(avg_wer_2)
  print(avg_wer_3)
  print(avg_wer_4)
  print(avg_wer_5)