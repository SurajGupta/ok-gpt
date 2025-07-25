import json

from recorder import live_speech, calibrate_decibles

calibrate_decibles(65)

# wakeup_words = []

# for i in range(10):
#     print("Please say the wakeup keyphrase")
#     for phrase in live_speech():
#         print(f"Heard '{str(phrase)}'\n")
#         wakeup_words.append(phrase)
#         break

# with open("wakeup_words.json", "w") as f:
#     json.dump(list(set(wakeup_words)), f)

# print("Recognition finished")

