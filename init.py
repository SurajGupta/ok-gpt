import json

from recorder import calibrate_decibles, listen_for_and_transcribe_potential_wake_word

# calibrate_decibles(65)
listen_for_and_transcribe_potential_wake_word(65)

# wakeup_words = []

# for i in range(10):
#     print("Please say the wakeup keyphrase")
#     generator = live_speech()
#     try:
#         for phrase in generator:
#             print(f"Heard '{str(phrase)}'\n")
#             wakeup_words.append(phrase)
#             break
#     finally:
#         generator.close()

# with open("wakeup_words.json", "w") as f:
#     json.dump(list(set(wakeup_words)), f)

# print("Recognition finished")

