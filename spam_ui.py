import pickle
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button

# Load trained spam detection model
with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

class SpamDetectorApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=10)

        self.label = Label(text="📩 Enter Email Content:", font_size=20)
        self.layout.add_widget(self.label)

        self.input_text = TextInput(multiline=False, hint_text="Type your email here...", font_size=18)
        self.layout.add_widget(self.input_text)

        self.button = Button(text="Check for Spam", font_size=18, background_color=(0, 0.5, 1, 1))
        self.button.bind(on_press=self.check_spam)
        self.layout.add_widget(self.button)

        self.result_label = Label(text="", font_size=22, bold=True)
        self.layout.add_widget(self.result_label)

        return self.layout

    def check_spam(self, instance):
        email_content = self.input_text.text
        prediction = model.predict([email_content])[0]
        self.result_label.text = "🚨 SPAM" if prediction else "✅ NOT SPAM"

if __name__ == "__main__":
    SpamDetectorApp().run()
