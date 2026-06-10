import os
import requests
import sys

def play_beep():
    """Plays a system beep."""
    if sys.platform == 'darwin':
        os.system('say -v Fred "Alert"') # Text-to-speech on Mac
    else:
        # Default beep for other OS (simple ASCII bell)
        print('\a')

class DiscordNotifier:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url

    def send_alert(self, title, message, color=0x00ff00):
        """
        Sends a rich embed alert to Discord.
        Color: 0x00ff00 (Green), 0xff0000 (Red)
        """
        if not self.webhook_url:
             print("[MOCK DISCORD] Creds missing. Alert:", title)
             return

        payload = {
            "embeds": [
                {
                    "title": title,
                    "description": message,
                    "color": color
                }
            ]
        }
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            if response.status_code == 204:
                print("Discord alert sent.")
            else:
                print(f"Failed to send Discord alert: {response.text}")
        except Exception as e:
            print(f"Error sending Discord alert: {e}")

    def send_chart(self, image_buffer, comment="Chart Analysis"):
        """
        Uploads a chart image to Discord.
        """
        if not self.webhook_url:
            print("[MOCK DISCORD] Creds missing. Chart not sent.")
            return

        files = {
            'file': ('chart.png', image_buffer, 'image/png')
        }
        data = {
            'content': comment
        }
        
        try:
            response = requests.post(self.webhook_url, data=data, files=files, timeout=20)
            if response.status_code in [200, 204]:
                print("Discord chart sent.")
            else:
                print(f"Failed to send Discord chart: {response.text}")
        except Exception as e:
            print(f"Error sending Discord chart: {e}")
