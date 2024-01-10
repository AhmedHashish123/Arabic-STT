from jiwer import wer

def calculate_wer(fleurs_test, model_1, model_2, model_3):
  total_wer_1 = 0
  total_wer_2 = 0
  total_wer_3 = 0

  for i in range(len(fleurs_test)):
    reference = fleurs_test[i]['sentence']
    hypothesis_1 = model_1.transcribe(fleurs_test[i]['path'])
    hypothesis_2 = model_2.transcribe(fleurs_test[i]['path'])
    hypothesis_3 = model_3.transcribe(fleurs_test[i]['path'])
    wer_1 = wer(reference, hypothesis_1)
    wer_2 = wer(reference, hypothesis_2)
    wer_3 = wer(reference, hypothesis_3)
    total_wer_1 += wer_1
    total_wer_2 += wer_2
    total_wer_3 += wer_3

  avg_wer_1 = total_wer_1 / len(fleurs_test)
  avg_wer_2 = total_wer_2 / len(fleurs_test)
  avg_wer_3 = total_wer_3 / len(fleurs_test)

  print(avg_wer_1)
  print(avg_wer_2)
  print(avg_wer_3)