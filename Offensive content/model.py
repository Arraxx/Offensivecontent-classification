import pandas as pd
import pickle 

df = pd.read_csv('Offensive_Content_data.csv', sep=',')

#select relavant columns
tweet_df = df[['tweet','label']]

sentiment_label = tweet_df.label

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tweets = tweet_df.tweet.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweets)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(tweets)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)
# Build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding

embedding_vector_length = 32
model = Sequential() 
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200) )
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  
print(model.summary())

history = model.fit(padded_sequence,sentiment_label,validation_split=0.2, epochs=2, batch_size=32)


# Serialize Model to JSON
model_json = model.to_json()

# Save Model to File
with open("model1.json", 'w') as json_file:
	json_file.write(model_json)

# Save Model Weights
model.save_weights("model1.h5")


print("Model Saved")

with open('tokenizer1.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Tokenizer Saved")





