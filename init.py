import json

from recorder import calibrate_decibles, listen_for_and_transcribe_potential_wake_word

# calibrate_decibles(65)

wakeup_words = []

generator = listen_for_and_transcribe_potential_wake_word(65, verbose = True, print_sample_number_when_verbose= True)

try:
    for i in range(10):
        # print("Please say the wakeup keyphrase")
        # this will block until live_speech yields a phrase
        phrase = next(generator)  
        # print(f"Heard '{phrase}'\n")
        wakeup_words.append(phrase)
finally:
    # Now we tear down the generator (runs its finally:)
    generator.close()


# with open("wakeup_words.json", "w") as f:
#     json.dump(list(set(wakeup_words)), f)

# print("Recognition finished")

