import json

from recorder import calibrate_decibles, establish_wake_words, wait_for_wake_words

# calibrate_decibles(65)
print ("waiting for wake word...")
wait_for_wake_words(65, decibles_that_indicate_speech = 50)
print ("wake word found!")
