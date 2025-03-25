from gtts import gTTS
import os

def create_test_speech():
    try:
        text = """
        Welcome to this test recording.
        We are creating a short podcast segment.
        This will help us test the video generation system.
        Each sentence should become its own segment.
        The segmentation algorithm will process this naturally.
        Let's see how the enhancement process works.
        We should see some interesting visual interpretations.
        This is nearing the end of our test.
        Thank you for listening to this test.
        """
        
        print("Generating speech from text...")
        tts = gTTS(text=text, lang='en', slow=False)
        
        print("Saving to short_test.wav...")
        tts.save("short_test.wav")
        
        print("Test audio file created successfully!")
        
    except Exception as e:
        print(f"Error creating test speech: {str(e)}")
        print("Note: This requires an internet connection as it uses Google's TTS service")

if __name__ == '__main__':
    create_test_speech()