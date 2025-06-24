import os
import pandas as pd
from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()
API_ID = int(os.getenv("TELEGRAM_API_ID"))
API_HASH = os.getenv("TELEGRAM_API_HASH")
SESSION = os.getenv("TELEGRAM_SESSION_NAME")

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
IMG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images'))
CHANNELS_FILE = os.path.join(DATA_DIR, "channels_to_crawl.txt")
OUTPUT_FILE = os.path.join(DATA_DIR, "raw_telegram_data.csv")
N_MESSAGES = 300  # Change as needed

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

def load_channels(file_path):
    with open(file_path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

async def fetch_channel_messages(client, channel, limit=N_MESSAGES):
    messages = []
    async for msg in client.iter_messages(channel, limit=limit):
        content = msg.text or ""
        sender = None
        try:
            sender = await msg.get_sender()
            sender = getattr(sender, 'username', None) or getattr(sender, 'first_name', None)
        except Exception:
            pass
        photo_path = None
        if msg.media and isinstance(msg.media, MessageMediaPhoto):
            photo_path = os.path.join(IMG_DIR, f"{channel}_{msg.id}.jpg")
            await msg.download_media(photo_path)
        messages.append({
            "msg_id": msg.id,
            "channel": channel,
            "sender": sender,
            "date": msg.date.isoformat(),
            "text": content,
            "photo": photo_path,
        })
    return messages

def main():
    import asyncio
    client = TelegramClient(SESSION, API_ID, API_HASH)
    channels = load_channels(CHANNELS_FILE)
    all_msgs = []

    async def run_all():
        await client.start()
        for channel in tqdm(channels, desc="Channels"):
            print(f"Fetching from: {channel}")
            msgs = await fetch_channel_messages(client, channel)
            all_msgs.extend(msgs)
        await client.disconnect()

    asyncio.run(run_all())
    df = pd.DataFrame(all_msgs)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} messages to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()