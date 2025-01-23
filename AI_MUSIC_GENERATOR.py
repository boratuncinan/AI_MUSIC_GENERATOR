#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, redirect
import spotipy
from spotipy.oauth2 import SpotifyOAuth

app = Flask(__name__)

CLIENT_ID = '**'
CLIENT_SECRET = '**'
REDIRECT_URI = 'http://localhost:5700/callback'
SCOPE = 'user-library-read'

sp_oauth = SpotifyOAuth(client_id=CLIENT_ID,
                        client_secret=CLIENT_SECRET,
                        redirect_uri=REDIRECT_URI,
                        scope=SCOPE)

@app.route('/')
def home():
    auth_url = sp_oauth.get_authorize_url()
    return f'<a href="{auth_url}">Login with Spotify</a>'

@app.route('/callback')
def callback():
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    access_token = token_info['access_token']
    return f"Access token: {access_token}"

if __name__ == '__main__':
    app.run(host='localhost', port=5700)


# In[ ]:


import urllib.parse

CLIENT_ID = '**'
REDIRECT_URI = 'http://localhost:5700/callback'
SCOPE = 'user-library-read'  
STATE = 'choculitis' 

auth_url = (
    "https://accounts.spotify.com/authorize?"
    f"response_type=code&client_id={CLIENT_ID}"
    f"&scope={urllib.parse.quote(SCOPE)}"
    f"&redirect_uri={urllib.parse.quote(REDIRECT_URI)}"
    f"&state={STATE}"
)

print("Authorization URL:", auth_url)


# In[ ]:


import os
import re
import requests

access_token = "**"

headers = {
    "Authorization": f"Bearer {access_token}"
}

def sanitize_filename(filename):

    return re.sub(r'[\/:*?"<>|]', '', filename)

def get_artist_id(artist_name):

    search_url = "https://api.spotify.com/v1/search"
    params = {
        "q": artist_name,
        "type": "artist",
        "limit": 1  
    }
    response = requests.get(search_url, headers=headers, params=params)

    if response.status_code != 200:
        print(f"API hatası: {response.status_code}")
        return None
    
    result = response.json()

    if 'artists' not in result or len(result['artists']['items']) == 0:
        print(f"Sanatçı '{artist_name}' bulunamadı.")
        return None

    artist_id = result['artists']['items'][0]['id'] 
    return artist_id

def get_albums(artist_id):

    albums_url = f"https://api.spotify.com/v1/artists/{artist_id}/albums"
    params = {
        "album_type": "album",
        "limit": 50 
    }
    albums = []
    response = requests.get(albums_url, headers=headers, params=params)
    results = response.json()
    
    albums.extend(results['items'])
    
    while results['next']:
        response = requests.get(results['next'], headers=headers)
        results = response.json()
        albums.extend(results['items'])
    
    return albums

def get_tracks(album_id):

    tracks_url = f"https://api.spotify.com/v1/albums/{album_id}/tracks"
    tracks = []
    response = requests.get(tracks_url, headers=headers)
    results = response.json()
    
    tracks.extend(results['items'])
    
    while results['next']:
        response = requests.get(results['next'], headers=headers)
        results = response.json()
        tracks.extend(results['items'])
    
    return tracks

def get_all_songs_by_artist(artist_name):

    artist_id = get_artist_id(artist_name)
    
    if not artist_id:
        return []

    albums = get_albums(artist_id)
    
    all_tracks = []
    for album in albums:
        album_tracks = get_tracks(album['id'])
        for track in album_tracks:
            all_tracks.append({
                'album_name': album['name'],
                'track_name': track['name'],
                'track_id': track['id'],
                'track_duration_ms': track['duration_ms'],
                'preview_url': track['preview_url']
            })
    
    return all_tracks

def download_song(preview_url, track_name, download_folder):

    if preview_url:
        response = requests.get(preview_url)
        if response.status_code == 200:
            safe_track_name = sanitize_filename(track_name)  
            file_name = f"{safe_track_name}.mp3"
            file_path = os.path.join(download_folder, file_name)
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"{file_name} downloaded and {download_folder} saved to that path.")
        else:
            print(f"Song did not downloaded: {track_name}")
    else:
        print(f"No preview URL: {track_name}")

if __name__ == "__main__":
    artist_name = ""  
    download_folder = r"**"  

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    all_songs = get_all_songs_by_artist(artist_name)
    
    if all_songs:
        for idx, song in enumerate(all_songs):
            print(f"{idx + 1}. {song['track_name']} (Album: {song['album_name']})")
            
            download_song(song['preview_url'], song['track_name'], download_folder)
    else:
        print("Artist does not found")


# In[ ]:


from pydub import AudioSegment
from pydub.utils import which

AudioSegment.converter = which(r"**")


# In[ ]:


import os
import subprocess

def convert_mp3_to_wav(input_folder, output_folder, ffmpeg_path):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".mp3"):
            mp3_file_path = os.path.join(input_folder, file_name)
            wav_file_name = file_name.replace(".mp3", ".wav")
            wav_file_path = os.path.join(output_folder, wav_file_name)

            command = [ffmpeg_path, "-i", mp3_file_path, wav_file_path]

            try:
                subprocess.run(command, check=True)
                print(f"Transformed: {file_name} -> {wav_file_name}")
            except subprocess.CalledProcessError as e:
                print(f"Error: {file_name} not transformed. Error: {e}")

input_folder = r"**" 
output_folder = r"**"  
ffmpeg_path = r"**"  

convert_mp3_to_wav(input_folder, output_folder, ffmpeg_path)


# In[ ]:


import librosa
import numpy as np
import pandas as pd
import os

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    features = {
        "file_name": file_path,
        "tempo": tempo,
        **{f"mfcc_{i+1}": np.mean(mfccs[i]) for i in range(13)},
        **{f"chroma_{i+1}": np.mean(chroma[i]) for i in range(12)},
        "zcr": np.mean(zcr),
        **{f"spectral_contrast_{i+1}": np.mean(spectral_contrast[i]) for i in range(spectral_contrast.shape[0])}
    }
    
    return features

def process_dataset(folder_path):
    feature_list = []
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            print(f"Processing file: {file}")
            features = extract_features(file_path)
            feature_list.append(features)
    
    df = pd.DataFrame(feature_list)
    
    df["tempo"] = df["tempo"].apply(lambda x: float(str(x).strip("[]")))
    df["file_name"] = df["file_name"].apply(lambda x: os.path.basename(x))
    
    return df

dataset_folder = r"**"

features_df = process_dataset(dataset_folder)

output_file = "cleaned_extracted_features_classic.csv"
features_df.to_csv(output_file, index=False)

print(f"Cleaned features '{output_file}' saved to that path")


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import os

genre_csv_paths = {
    "classic": r"**.csv",
    "jazz": r"**.csv",
    "pop": r"**.csv",
    "rap": r"**.csv",
    "rock": r"**.csv",
}

def train_and_save_genre_classifier():
    try:

        combined_data = pd.DataFrame()
        for genre, path in genre_csv_paths.items():
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue
            data = pd.read_csv(path)
            if 'file_name' in data.columns:
                data = data.drop(columns=['file_name'], errors='ignore')
            data['genre'] = genre  
            combined_data = pd.concat([combined_data, data], ignore_index=True)

        if combined_data.empty:
            print("Error: No data loaded from the CSV files.")
            return

        print(f"Loaded data: {combined_data.shape} rows and columns")

        feature_columns = [col for col in combined_data.columns if col != 'genre']
        X = combined_data[feature_columns]
        y = combined_data['genre']

        print(f"Feature columns: {len(feature_columns)}")
        print(f"Labels distribution: {y.value_counts()}")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        classifier = RandomForestClassifier(random_state=42)
        classifier.fit(X_scaled, y)

        save_path = r"**\genre_classifier.pkl"
        scaler_path = r"**\scaler.pkl"
        with open(save_path, 'wb') as file:
            pickle.dump(classifier, file)
        with open(scaler_path, 'wb') as file:
            pickle.dump(scaler, file)

        print(f"Genre classifier saved at: {save_path}")
        print(f"Scaler saved at: {scaler_path}")
    except Exception as e:
        print(f"Error during training and saving: {e}")

if __name__ == "__main__":
    train_and_save_genre_classifier()


# In[ ]:


import os
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler
import numpy as np

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.0, stddev=1.0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def train_and_save_vae(csv_path, output_dir, genre_name, latent_dim=16, epochs=50, batch_size=32):

    data = pd.read_csv(csv_path).drop(columns=['file_name'], errors='ignore')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    input_dim = scaled_data.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    z_mean = Dense(latent_dim, name='z_mean')(encoded)
    z_log_var = Dense(latent_dim, name='z_log_var')(encoded)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    decoded = Dense(128, activation='relu')(latent_inputs)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    encoder = Model(input_layer, [z_mean, z_log_var], name='encoder')
    decoder = Model(latent_inputs, output_layer, name='decoder')

    vae_output = decoder(z)
    vae = Model(input_layer, vae_output, name='vae')

    reconstruction_loss = K.sum(K.binary_crossentropy(input_layer, vae_output), axis=-1)
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae.add_loss(K.mean(reconstruction_loss + kl_loss))
    vae.compile(optimizer='adam')

    vae.fit(scaled_data, scaled_data, epochs=epochs, batch_size=batch_size, verbose=1)

    model_dir = os.path.join(output_dir, genre_name)
    os.makedirs(model_dir, exist_ok=True)
    vae_path = os.path.join(model_dir, f"vae_model_{genre_name}")
    encoder_path = os.path.join(model_dir, f"vae_encoder_{genre_name}")
    decoder_path = os.path.join(model_dir, f"vae_decoder_{genre_name}")

    vae.save(vae_path, save_format='tf')
    encoder.save(encoder_path, save_format='tf')
    decoder.save(decoder_path, save_format='tf')

    print(f"Models saved successfully for {genre_name}:\n"
          f"VAE -> {vae_path}\nEncoder -> {encoder_path}\nDecoder -> {decoder_path}")

genres = {
    "classic": r"**.csv",
    "jazz": r"**.csv",
    "pop": r"**.csv",
    "rap": r"**.csv",
    "rock": r"**.csv"
}
output_dir = r"**"

for genre_name, csv_path in genres.items():
    print(f"Training and saving VAE for genre: {genre_name}")
    train_and_save_vae(csv_path, output_dir, genre_name)


# In[ ]:


def generate_new_features(encoder_path, decoder_path, input_features):
    try:
        encoder = tf.keras.models.load_model(encoder_path, custom_objects={"sampling": sampling})
        decoder = tf.keras.models.load_model(decoder_path, custom_objects={"sampling": sampling})
        
        latent_vector = encoder.predict(input_features.reshape(1, -1))[2]
        generated_features = decoder.predict(latent_vector)
        
        print("Features generated successfully!")
        return generated_features.flatten()
    except Exception as e:
        print(f"Error generating features: {e}")
        return None

print("Feature generation function defined successfully!")


# In[ ]:


import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import os
import re
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import tensorflow as tf
from scipy.spatial.distance import euclidean
import json
from pygame import mixer

output_folder = r"**"
genre_classifier_path = r"**genre_classifier.pkl"
scaler_path = r"**\scaler.pkl"
vae_models_path = r"**"
genre_csv_paths = {
    "classic": r"**.csv",
    "jazz": r"**.csv",
    "pop": r"**.csv",
    "rap": r"**.csv",
    "rock": r"**.csv",
}

favorites_file = "favorites.json"
playlists_file = "playlists.json"

FEATURE_DESCRIPTIONS = {
    "tempo": "The speed of the music, measured in beats per minute (BPM).",
    "mfcc": "Mel-Frequency Cepstral Coefficients (MFCC) describe the timbre of the audio.",
    "chroma": "Chroma features represent the pitch content of the audio.",
    "zcr": "Zero Crossing Rate (ZCR) counts how often the signal changes sign, indicating noisiness.",
    "spectral_contrast": "Spectral contrast measures the difference in amplitude between peaks and valleys in the spectrum."
}

if os.path.exists(favorites_file):
    with open(favorites_file, "r") as file:
        favorites = json.load(file)
else:
    favorites = []

if os.path.exists(playlists_file):
    with open(playlists_file, "r") as file:
        playlists = json.load(file)
else:
    playlists = {}

def save_favorites():
    with open(favorites_file, "w") as file:
        json.dump(favorites, file)

def save_playlists():
    with open(playlists_file, "w") as file:
        json.dump(playlists, file)

def predict_genre(features):
    try:
        with open(genre_classifier_path, 'rb') as file:
            classifier = pickle.load(file)
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)

        feature_vector = scaler.transform([list(features.values())])
        predicted_genre = classifier.predict(feature_vector)[0]
        return predicted_genre
    except Exception as e:
        print(f"Error predicting genre: {e}")
        return None

def get_next_song_name(output_folder):
    existing_files = os.listdir(output_folder)
    song_numbers = [int(re.match(r"song#(\d+)\.wav", file).group(1))
                    for file in existing_files if re.match(r"song#(\d+)\.wav", file)]
    next_number = max(song_numbers) + 1 if song_numbers else 1
    return f"song#{next_number}.wav"

def upload_and_process_song(status_label, feature_tree, btn_favorite, btn_playlist, btn_play, btn_stop):
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if not file_path:
        return
    try:
        file_name = os.path.basename(file_path)

        uploaded_song_features = extract_features(file_path)

        predicted_genre = predict_genre(uploaded_song_features)

        encoder_path = f"{vae_models_path}/{predicted_genre}/vae_encoder_{predicted_genre}"
        decoder_path = f"{vae_models_path}/{predicted_genre}/vae_decoder_{predicted_genre}"
        vae_encoder = tf.keras.models.load_model(encoder_path, compile=False)
        vae_decoder = tf.keras.models.load_model(decoder_path, compile=False)

        result_message = find_and_create_song(uploaded_song_features, output_folder, file_path)

        display_features(uploaded_song_features, feature_tree, status_label, result_message, file_name, predicted_genre)

        btn_favorite.config(state="normal")
        btn_playlist.config(state="normal")
        btn_play.config(state="normal")
        btn_stop.config(state="normal")
        btn_favorite.song_name = result_message.split(" at ")[-1]  
        btn_playlist.song_name = btn_favorite.song_name
        btn_play.song_name = btn_favorite.song_name
    except Exception as e:
        status_label.config(text=f"An error occurred: {e}", foreground="red")
        
def create_audio(file_path, output_path, manipulation_params=None, add_noise=True):

    try:
        pitch_shift = manipulation_params.get("pitch_shift", 2) if manipulation_params else 2
        time_stretch = manipulation_params.get("time_stretch", 1.2) if manipulation_params else 1.2

        y, sr = librosa.load(file_path, sr=None)

        if pitch_shift != 0:  
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)

        if time_stretch != 1:  
            y = librosa.effects.time_stretch(y, rate=time_stretch)

        if add_noise:
            noise = np.random.normal(0, 0.01, y.shape)
            y = y + noise

        sf.write(output_path, y, sr)
        return f"Created audio saved at {output_path}"
    except Exception as e:
        return f"Error during audio creation: {e}"

def display_features(features, tree, status_label, result_message, file_name, predicted_genre):
    for item in tree.get_children():
        tree.delete(item)

    for key, value in features.items():
        tree.insert("", "end", values=(key, f"{value:.4f}" if isinstance(value, (float, int)) else value))

    tree.insert("", "end", values=("Uploaded Song", file_name))
    tree.insert("", "end", values=("Predicted Genre", predicted_genre))

    tree.insert("", "end", values=("Creation Info", result_message))
    status_label.config(text=result_message, foreground="green")

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        features = {
            "tempo": float(tempo),
            **{f"mfcc_{i+1}": np.mean(mfccs[i]) for i in range(13)},
            **{f"chroma_{i+1}": np.mean(chroma[i]) for i in range(12)},
            "zcr": np.mean(zcr),
            **{f"spectral_contrast_{i+1}": np.mean(spectral_contrast[i]) for i in range(spectral_contrast.shape[0])}
        }
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        raise
        
def generate_new_features(encoder_path, decoder_path, input_features):
    try:
        encoder = tf.keras.models.load_model(encoder_path, compile=False)
        decoder = tf.keras.models.load_model(decoder_path, compile=False)

        input_features = np.array(input_features).reshape(1, -1)
        latent_vector = encoder.predict(input_features)[0]
        latent_vector_reshaped = latent_vector.reshape(1, -1)
        generated_features = decoder.predict(latent_vector_reshaped)
        return generated_features.flatten()
    except Exception as e:
        print(f"Error generating features: {e}")
        return None

def show_help():
    help_window = tk.Toplevel()
    help_window.title("Feature Descriptions")

    help_text = "\n".join([
        f"{key.capitalize()}: {value}"
        for key, value in FEATURE_DESCRIPTIONS.items()
    ])

    text_widget = tk.Text(help_window, wrap="word", font=("Helvetica", 12), height=10, width=60)
    text_widget.insert("1.0", help_text)
    text_widget.config(state="disabled")
    text_widget.pack(fill="both", expand=True, padx=10, pady=10)

    btn_close = tk.Button(help_window, text="Close", command=help_window.destroy, font=("Helvetica", 12))
    btn_close.pack(pady=5)

    help_window.update_idletasks() 
    help_window.geometry(f"{help_window.winfo_width()}x{help_window.winfo_height()}")


def add_to_favorites(song_name):
    if song_name not in favorites:
        favorites.append(song_name)
        save_favorites()
        messagebox.showinfo("Favorites", f"'{song_name}' added to favorites.")
    else:
        messagebox.showinfo("Favorites", f"'{song_name}' is already in favorites.")

def remove_from_favorites(song_name):
    if song_name in favorites:
        favorites.remove(song_name)
        save_favorites()
        messagebox.showinfo("Favorites", f"'{song_name}' removed from favorites.")
    else:
        messagebox.showinfo("Favorites", f"'{song_name}' is not in favorites.")

def add_to_playlist(song_name):
    playlist_name = simpledialog.askstring("Playlist Name", "Enter the name of the playlist:")
    if not playlist_name:
        return

    if playlist_name not in playlists:
        playlists[playlist_name] = []

    playlists[playlist_name].append(song_name)
    save_playlists()

    messagebox.showinfo("Playlists", f"'{song_name}' added to playlist '{playlist_name}'.")

def remove_from_playlist(playlist_name, song_name):
    if playlist_name in playlists and song_name in playlists[playlist_name]:
        playlists[playlist_name].remove(song_name)
        save_playlists()
        messagebox.showinfo("Playlists", f"'{song_name}' removed from playlist '{playlist_name}'.")
    else:
        messagebox.showinfo("Playlists", f"'{song_name}' is not in playlist '{playlist_name}'.")

def play_song(song_name):
    try:
        mixer.init()
        mixer.music.load(song_name)
        mixer.music.play()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to play song: {e}")

def stop_song():
    try:
        mixer.music.stop()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to stop song: {e}")

def close_app(root):
    try:
        mixer.music.stop()
        root.destroy()
    except Exception as e:
        root.destroy()

def show_playlists():
    playlists_window = tk.Toplevel()
    playlists_window.title("Playlists")
    playlists_window.geometry("800x600")

    listbox = tk.Listbox(playlists_window, font=("Helvetica", 14))
    listbox.pack(fill="both", expand=True)

    def remove_playlist():
        selected = listbox.curselection()
        if selected:
            playlist_name = listbox.get(selected[0])
            if messagebox.askyesno("Confirm", f"Are you sure you want to delete the playlist '{playlist_name}'?"):
                del playlists[playlist_name]
                save_playlists()
                listbox.delete(selected[0])

    def display_playlist_songs(playlist_name):
        playlist_songs_window = tk.Toplevel(playlists_window)
        playlist_songs_window.title(f"Songs in {playlist_name}")
        playlist_songs_window.geometry("800x600")

        songs_listbox = tk.Listbox(playlist_songs_window, font=("Helvetica", 14))
        songs_listbox.pack(fill="both", expand=True)

        for song in playlists.get(playlist_name, []):
            songs_listbox.insert(tk.END, song)

        def play_selected_song():
            selected = songs_listbox.curselection()
            if selected:
                song_name = songs_listbox.get(selected[0])
                play_song(song_name)

        def stop_selected_song():
            stop_song()

        def remove_selected_song():
            selected = songs_listbox.curselection()
            if selected:
                song_name = songs_listbox.get(selected[0])
                if messagebox.askyesno("Confirm", f"Are you sure you want to remove '{song_name}' from the playlist?"):
                    remove_from_playlist(playlist_name, song_name)
                    songs_listbox.delete(selected[0])

        btn_play_song = tk.Button(playlist_songs_window, text="Play Song", command=play_selected_song, font=("Helvetica", 12))
        btn_play_song.pack(side="left", padx=5, pady=5)

        btn_stop_song = tk.Button(playlist_songs_window, text="Stop Song", command=stop_selected_song, font=("Helvetica", 12))
        btn_stop_song.pack(side="left", padx=5, pady=5)

        btn_remove_song = tk.Button(playlist_songs_window, text="Remove Song", command=remove_selected_song, font=("Helvetica", 12))
        btn_remove_song.pack(side="left", padx=5, pady=5)

    for playlist in playlists.keys():
        listbox.insert(tk.END, playlist)

    def on_select(event):
        selected = listbox.curselection()
        if selected:
            playlist_name = listbox.get(selected[0])
            display_playlist_songs(playlist_name)

    listbox.bind('<<ListboxSelect>>', on_select)

    btn_remove_playlist = tk.Button(playlists_window, text="Remove Playlist", command=remove_playlist, font=("Helvetica", 12))
    btn_remove_playlist.pack(pady=5)

def show_favorites():
    favorites_window = tk.Toplevel()
    favorites_window.title("Favorites")
    favorites_window.geometry("800x600")

    listbox = tk.Listbox(favorites_window, font=("Helvetica", 14))
    listbox.pack(fill="both", expand=True)

    for song in favorites:
        listbox.insert(tk.END, song)

    def play_selected_song():
        selected = listbox.curselection()
        if selected:
            song_name = listbox.get(selected[0])
            play_song(song_name)

    def stop_selected_song():
        stop_song()

    def remove_selected_song():
        selected = listbox.curselection()
        if selected:
            song_name = listbox.get(selected[0])
            if messagebox.askyesno("Confirm", f"Are you sure you want to remove '{song_name}' from favorites?"):
                remove_from_favorites(song_name)
                listbox.delete(selected[0])

    btn_play_song = tk.Button(favorites_window, text="Play Song", command=play_selected_song, font=("Helvetica", 12))
    btn_play_song.pack(side="left", padx=5, pady=5)

    btn_stop_song = tk.Button(favorites_window, text="Stop Song", command=stop_selected_song, font=("Helvetica", 12))
    btn_stop_song.pack(side="left", padx=5, pady=5)

    btn_remove_song = tk.Button(favorites_window, text="Remove Song", command=remove_selected_song, font=("Helvetica", 12))
    btn_remove_song.pack(side="left", padx=5, pady=5)

messagebox.showinfo = lambda title, message: print(f"INFO: {title} - {message}")

def main_gui():
    root = tk.Tk()
    root.title("AI Music Generator")
    root.attributes("-fullscreen", True)

    title_label = tk.Label(root, text="AI Music Generator", font=("Helvetica", 20, "bold"), bg="#282c34", fg="white")
    title_label.pack(fill="x", pady=10)

    frame_controls = tk.Frame(root)
    frame_controls.pack(fill="x", padx=10, pady=10)

    frame_controls_inner = tk.Frame(frame_controls)
    frame_controls_inner.pack(anchor="center")

    btn_favorite = tk.Button(frame_controls_inner, text="Add to Favorites", state="disabled", font=("Helvetica", 12), command=lambda: add_to_favorites(btn_favorite.song_name))
    btn_favorite.pack(side="left", padx=5, pady=5)

    btn_playlist = tk.Button(frame_controls_inner, text="Add to Playlist", state="disabled", font=("Helvetica", 12), command=lambda: add_to_playlist(btn_playlist.song_name))
    btn_playlist.pack(side="left", padx=5, pady=5)

    btn_upload = tk.Button(frame_controls_inner, text="Upload & Process", font=("Helvetica", 12), command=lambda: upload_and_process_song(status_label, feature_tree, btn_favorite, btn_playlist, btn_play, btn_stop))
    btn_upload.pack(side="left", padx=5, pady=5)

    btn_favorites = tk.Button(frame_controls_inner, text="Show Favorites", font=("Helvetica", 12), command=show_favorites)
    btn_favorites.pack(side="left", padx=5, pady=5)

    btn_playlists = tk.Button(frame_controls_inner, text="Show Playlists", font=("Helvetica", 12), command=show_playlists)
    btn_playlists.pack(side="left", padx=5, pady=5)

    btn_help = tk.Button(frame_controls_inner, text="Help", font=("Helvetica", 12), command=show_help)
    btn_help.pack(side="left", padx=5, pady=5)

    frame_main = tk.Frame(root)
    frame_main.pack(fill="both", expand=True, padx=10, pady=10)

    status_label = tk.Label(frame_main, text="Status: Waiting for input", fg="blue", font=("Helvetica", 14))
    status_label.pack(pady=5)

    feature_tree = ttk.Treeview(frame_main, columns=("Feature", "Value"), show="headings", height=20)
    feature_tree.heading("Feature", text="Feature")
    feature_tree.heading("Value", text="Value")
    feature_tree.column("Feature", anchor="center", width=200)
    feature_tree.column("Value", anchor="center", width=200)
    feature_tree.tag_configure("font", font=("Helvetica", 14))
    feature_tree.pack(fill="both", expand=True)

    frame_bottom = tk.Frame(root, bg="#282c34")
    frame_bottom.pack(fill="x", pady=10)

    btn_play = tk.Button(frame_bottom, text="▶ Play Song", state="disabled", command=lambda: play_song(btn_play.song_name), bg="#61afef", fg="white", font=("Helvetica", 16, "bold"))
    btn_play.pack(side="left", padx=10, pady=10, expand=True)

    btn_stop = tk.Button(frame_bottom, text="⏹ Stop Song", state="disabled", command=stop_song, bg="#e06c75", fg="white", font=("Helvetica", 16, "bold"))
    btn_stop.pack(side="left", padx=10, pady=10, expand=True)

    btn_close = tk.Button(frame_bottom, text="❌ Close App", command=lambda: close_app(root), bg="#98c379", fg="white", font=("Helvetica", 16, "bold"))
    btn_close.pack(side="left", padx=10, pady=10, expand=True)

    root.mainloop()

if __name__ == "__main__":
    main_gui()

