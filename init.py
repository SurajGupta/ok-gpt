import json

from recorder import establish_wake_words, wait_for_wake_words

# calibrate_decibles(65)
print ("waiting for wake word...")
establish_wake_words()
print ("wake word found!")
