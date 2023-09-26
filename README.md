{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "id": "6ScJcc46tMyV"
      },
      "outputs": [],
      "source": [
        "import io\n",
        "import os\n",
        "import re\n",
        "import shutil\n",
        "import string\n",
        "import tensorflow as tf\n",
        "\n",
        "from tensorflow.keras import Sequential\n",
        "from tensorflow.keras import layers"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Download the IMDb Dataset\n",
        "\n",
        "We use the Large Movie Review Dataset through the tutorial. You will train a sentiment classifier model on this dataset and in the process learn embeddings from scratch."
      ],
      "metadata": {
        "id": "ki7Aifgctlc6"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "url = \"https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz\"\n",
        "\n",
        "dataset = tf.keras.utils.get_file(\"aclImdb_v1.tar.gz\", url,\n",
        "                                  untar=True, cache_dir='.',\n",
        "                                  cache_subdir='')\n",
        "\n",
        "dataset_dir = './aclImdb'\n",
        "os.listdir(dataset_dir)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tkAlpJ95tYDy",
        "outputId": "80cc2f25-cab6-4142-984a-a72ca5bfee9b"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['imdbEr.txt', 'README', 'train', 'test', 'imdb.vocab']"
            ]
          },
          "metadata": {},
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "dataset"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "3bxJGOhp9fbd",
        "outputId": "dee61ea1-47a1-4841-8dc2-7773f44cca2b"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'./aclImdb_v1'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "dataset_dir"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "1lV9hGzQ942k",
        "outputId": "d0c618d3-4f01-44e3-94fe-6aa63b59a886"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'./aclImdb'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_dir = os.path.join(dataset_dir, 'train')\n",
        "os.listdir(train_dir)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TzWfNOrfujvB",
        "outputId": "21c06c44-15df-458d-f014-fc9e63bc43ef"
      },
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['neg',\n",
              " 'unsup',\n",
              " 'urls_neg.txt',\n",
              " 'urls_pos.txt',\n",
              " 'urls_unsup.txt',\n",
              " 'unsupBow.feat',\n",
              " 'pos',\n",
              " 'labeledBow.feat']"
            ]
          },
          "metadata": {},
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "remove_dir = os.path.join(train_dir, 'unsup')\n",
        "shutil.rmtree(remove_dir)"
      ],
      "metadata": {
        "id": "Ux0RruM5umRC"
      },
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "batch_size = 1024\n",
        "seed = 123\n",
        "train_ds = tf.keras.utils.text_dataset_from_directory(\n",
        "    'aclImdb/train', batch_size=batch_size, validation_split=0.2,\n",
        "    subset='training', seed=seed)\n",
        "val_ds = tf.keras.utils.text_dataset_from_directory(\n",
        "    'aclImdb/train', batch_size=batch_size, validation_split=0.2,\n",
        "    subset='validation', seed=seed)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CfIagC-_up2q",
        "outputId": "5f6892cd-8266-4a92-c5c4-9c841e1185b9"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 25000 files belonging to 2 classes.\n",
            "Using 20000 files for training.\n",
            "Found 25000 files belonging to 2 classes.\n",
            "Using 5000 files for validation.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "for text_batch, label_batch in train_ds.take(1):\n",
        "  for i in range(5):\n",
        "    print(label_batch[i].numpy(), text_batch.numpy()[i])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FpVDHUnOutu6",
        "outputId": "80888725-1dc4-4cf6-d3ab-62d30e75b740"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0 b\"Oh My God! Please, for the love of all that is holy, Do Not Watch This Movie! It it 82 minutes of my life I will never get back. Sure, I could have stopped watching half way through. But I thought it might get better. It Didn't. Anyone who actually enjoyed this movie is one seriously sick and twisted individual. No wonder us Australians/New Zealanders have a terrible reputation when it comes to making movies. Everything about this movie is horrible, from the acting to the editing. I don't even normally write reviews on here, but in this case I'll make an exception. I only wish someone had of warned me before I hired this catastrophe\"\n",
            "1 b'This movie is SOOOO funny!!! The acting is WONDERFUL, the Ramones are sexy, the jokes are subtle, and the plot is just what every high schooler dreams of doing to his/her school. I absolutely loved the soundtrack as well as the carefully placed cynicism. If you like monty python, You will love this film. This movie is a tad bit \"grease\"esk (without all the annoying songs). The songs that are sung are likable; you might even find yourself singing these songs once the movie is through. This musical ranks number two in musicals to me (second next to the blues brothers). But please, do not think of it as a musical per say; seeing as how the songs are so likable, it is hard to tell a carefully choreographed scene is taking place. I think of this movie as more of a comedy with undertones of romance. You will be reminded of what it was like to be a rebellious teenager; needless to say, you will be reminiscing of your old high school days after seeing this film. Highly recommended for both the family (since it is a very youthful but also for adults since there are many jokes that are funnier with age and experience.'\n",
            "0 b\"Alex D. Linz replaces Macaulay Culkin as the central figure in the third movie in the Home Alone empire. Four industrial spies acquire a missile guidance system computer chip and smuggle it through an airport inside a remote controlled toy car. Because of baggage confusion, grouchy Mrs. Hess (Marian Seldes) gets the car. She gives it to her neighbor, Alex (Linz), just before the spies turn up. The spies rent a house in order to burglarize each house in the neighborhood until they locate the car. Home alone with the chicken pox, Alex calls 911 each time he spots a theft in progress, but the spies always manage to elude the police while Alex is accused of making prank calls. The spies finally turn their attentions toward Alex, unaware that he has rigged devices to cleverly booby-trap his entire house. Home Alone 3 wasn't horrible, but probably shouldn't have been made, you can't just replace Macauley Culkin, Joe Pesci, or Daniel Stern. Home Alone 3 had some funny parts, but I don't like when characters are changed in a movie series, view at own risk.\"\n",
            "0 b\"There's a good movie lurking here, but this isn't it. The basic idea is good: to explore the moral issues that would face a group of young survivors of the apocalypse. But the logic is so muddled that it's impossible to get involved.<br /><br />For example, our four heroes are (understandably) paranoid about catching the mysterious airborne contagion that's wiped out virtually all of mankind. Yet they wear surgical masks some times, not others. Some times they're fanatical about wiping down with bleach any area touched by an infected person. Other times, they seem completely unconcerned.<br /><br />Worse, after apparently surviving some weeks or months in this new kill-or-be-killed world, these people constantly behave like total newbs. They don't bother accumulating proper equipment, or food. They're forever running out of fuel in the middle of nowhere. They don't take elementary precautions when meeting strangers. And after wading through the rotting corpses of the entire human race, they're as squeamish as sheltered debutantes. You have to constantly wonder how they could have survived this long... and even if they did, why anyone would want to make a movie about them.<br /><br />So when these dweebs stop to agonize over the moral dimensions of their actions, it's impossible to take their soul-searching seriously. Their actions would first have to make some kind of minimal sense.<br /><br />On top of all this, we must contend with the dubious acting abilities of Chris Pine. His portrayal of an arrogant young James T Kirk might have seemed shrewd, when viewed in isolation. But in Carriers he plays on exactly that same note: arrogant and boneheaded. It's impossible not to suspect that this constitutes his entire dramatic range.<br /><br />On the positive side, the film *looks* excellent. It's got an over-sharp, saturated look that really suits the southwestern US locale. But that can't save the truly feeble writing nor the paper-thin (and annoying) characters. Even if you're a fan of the end-of-the-world genre, you should save yourself the agony of watching Carriers.\"\n",
            "0 b'I saw this movie at an actual movie theater (probably the $2.00 one) with my cousin and uncle. We were around 11 and 12, I guess, and really into scary movies. I remember being so excited to see it because my cool uncle let us pick the movie (and we probably never got to do that again!) and sooo disappointed afterwards!! Just boring and not scary. The only redeeming thing I can remember was Corky Pigeon from Silver Spoons, and that wasn\\'t all that great, just someone I recognized. I\\'ve seen bad movies before and this one has always stuck out in my mind as the worst. This was from what I can recall, one of the most boring, non-scary, waste of our collective $6, and a waste of film. I have read some of the reviews that say it is worth a watch and I say, \"Too each his own\", but I wouldn\\'t even bother. Not even so bad it\\'s good.'\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "AUTOTUNE = tf.data.AUTOTUNE\n",
        "\n",
        "train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)\n",
        "val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)"
      ],
      "metadata": {
        "id": "MAqrmi2au_u6"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Embed a 1,000 word vocabulary into 5 dimensions.\n",
        "embedding_layer = tf.keras.layers.Embedding(1000, 5)"
      ],
      "metadata": {
        "id": "JiVQ2gotvKZ6"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Testing the embedding layer (random embedding matrix) with some random word index\n",
        "result = embedding_layer(tf.constant([0, 2, 3]))\n",
        "result.numpy()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8W0tZr6ivMwq",
        "outputId": "54417929-a3c3-4410-df65-8827eab89917"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[ 0.03560449,  0.0315798 , -0.02038587,  0.02860259,  0.045079  ],\n",
              "       [ 0.01390843,  0.03567019, -0.03803892, -0.01430435, -0.04825154],\n",
              "       [-0.02644529, -0.01564109, -0.03266977, -0.04985544,  0.02334107]],\n",
              "      dtype=float32)"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Create a custom standardization function to strip HTML break tags '<br />'.\n",
        "def custom_standardization(input_data):\n",
        "  lowercase = tf.strings.lower(input_data)\n",
        "  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')\n",
        "  return tf.strings.regex_replace(stripped_html,\n",
        "                                  '[%s]' % re.escape(string.punctuation), '')"
      ],
      "metadata": {
        "id": "E9fkZrQ_vgs6"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "string.punctuation"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "YTKFjm9ABTdF",
        "outputId": "e35ee6be-037a-4f78-d08f-ff85c366df43"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'!\"#$%&\\'()*+,-./:;<=>?@[\\\\]^_`{|}~'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 26
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Vocabulary size and number of words in a sequence.\n",
        "vocab_size = 10000\n",
        "sequence_length = 100\n",
        "\n",
        "# Use the text vectorization layer to normalize, split, and map strings to\n",
        "# integers. Note that the layer uses the custom standardization defined above.\n",
        "# Set maximum_sequence length as all samples are not of the same length.\n",
        "vectorize_layer = layers.TextVectorization(\n",
        "    standardize=custom_standardization,\n",
        "    max_tokens=vocab_size,\n",
        "    output_mode='int',\n",
        "    output_sequence_length=sequence_length)\n",
        "\n",
        "# Make a text-only dataset (no labels) and call adapt to build the vocabulary.\n",
        "text_ds = train_ds.map(lambda x, y: x)\n",
        "vectorize_layer.adapt(text_ds)"
      ],
      "metadata": {
        "id": "OnognDnPwDIy"
      },
      "execution_count": 27,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "embedding_dim=16\n",
        "\n",
        "model = Sequential([\n",
        "  vectorize_layer,\n",
        "  layers.Embedding(vocab_size, embedding_dim, name=\"embedding\"),\n",
        "  layers.GlobalAveragePooling1D(),\n",
        "  layers.Dense(16, activation='relu'),\n",
        "  layers.Dense(1, activation='sigmoid')\n",
        "])"
      ],
      "metadata": {
        "id": "brfZSlMywZbR"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.compile(optimizer='adam',\n",
        "              loss='binary_crossentropy',\n",
        "              metrics=['accuracy'])"
      ],
      "metadata": {
        "id": "8bC9lpuewvHB"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.summary()"
      ],
      "metadata": {
        "id": "kB4h1c1GxOZ6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.fit(\n",
        "    train_ds,\n",
        "    validation_data=val_ds,\n",
        "    epochs=15)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5lguadzsxGph",
        "outputId": "d28e3fad-45e1-46b1-a07d-a279b53e8676"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/15\n",
            "20/20 [==============================] - 5s 205ms/step - loss: 2.2158 - accuracy: 0.5028 - val_loss: 1.5750 - val_accuracy: 0.4886\n",
            "Epoch 2/15\n",
            "20/20 [==============================] - 2s 80ms/step - loss: 1.4170 - accuracy: 0.5028 - val_loss: 1.3456 - val_accuracy: 0.4886\n",
            "Epoch 3/15\n",
            "20/20 [==============================] - 2s 128ms/step - loss: 1.2442 - accuracy: 0.5028 - val_loss: 1.2047 - val_accuracy: 0.4886\n",
            "Epoch 4/15\n",
            "20/20 [==============================] - 2s 82ms/step - loss: 1.1235 - accuracy: 0.5028 - val_loss: 1.0948 - val_accuracy: 0.4886\n",
            "Epoch 5/15\n",
            "20/20 [==============================] - 2s 82ms/step - loss: 1.0264 - accuracy: 0.5028 - val_loss: 1.0036 - val_accuracy: 0.4886\n",
            "Epoch 6/15\n",
            "20/20 [==============================] - 2s 81ms/step - loss: 0.9455 - accuracy: 0.5028 - val_loss: 0.9272 - val_accuracy: 0.4886\n",
            "Epoch 7/15\n",
            "20/20 [==============================] - 2s 81ms/step - loss: 0.8781 - accuracy: 0.5028 - val_loss: 0.8637 - val_accuracy: 0.4886\n",
            "Epoch 8/15\n",
            "20/20 [==============================] - 2s 80ms/step - loss: 0.8227 - accuracy: 0.5028 - val_loss: 0.8118 - val_accuracy: 0.4886\n",
            "Epoch 9/15\n",
            "20/20 [==============================] - 2s 80ms/step - loss: 0.7779 - accuracy: 0.5028 - val_loss: 0.7700 - val_accuracy: 0.4886\n",
            "Epoch 10/15\n",
            "20/20 [==============================] - 2s 101ms/step - loss: 0.7424 - accuracy: 0.5028 - val_loss: 0.7371 - val_accuracy: 0.4886\n",
            "Epoch 11/15\n",
            "20/20 [==============================] - 2s 80ms/step - loss: 0.7149 - accuracy: 0.5028 - val_loss: 0.7118 - val_accuracy: 0.4886\n",
            "Epoch 12/15\n",
            "20/20 [==============================] - 2s 80ms/step - loss: 0.6940 - accuracy: 0.5028 - val_loss: 0.6926 - val_accuracy: 0.4886\n",
            "Epoch 13/15\n",
            "20/20 [==============================] - 2s 80ms/step - loss: 0.6781 - accuracy: 0.5028 - val_loss: 0.6779 - val_accuracy: 0.4886\n",
            "Epoch 14/15\n",
            "20/20 [==============================] - 2s 79ms/step - loss: 0.6658 - accuracy: 0.5096 - val_loss: 0.6665 - val_accuracy: 0.5042\n",
            "Epoch 15/15\n",
            "20/20 [==============================] - 2s 80ms/step - loss: 0.6558 - accuracy: 0.5778 - val_loss: 0.6571 - val_accuracy: 0.6174\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7f012c3d1710>"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "weights = model.get_layer('embedding').get_weights()[0]\n",
        "vocab = vectorize_layer.get_vocabulary()"
      ],
      "metadata": {
        "id": "XUCbELbAxWsp"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "out_v = io.open('vectors.tsv', 'w', encoding='utf-8')\n",
        "out_m = io.open('metadata.tsv', 'w', encoding='utf-8')\n",
        "\n",
        "for index, word in enumerate(vocab):\n",
        "  if index == 0:\n",
        "    continue  # skip 0, it's padding.\n",
        "  vec = weights[index]\n",
        "  out_v.write('\\t'.join([str(x) for x in vec]) + \"\\n\")\n",
        "  out_m.write(word + \"\\n\")\n",
        "out_v.close()\n",
        "out_m.close()"
      ],
      "metadata": {
        "id": "lI3BZBJ-xZCp"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "[http://projector.tensorflow.org/](http://projector.tensorflow.org/)"
      ],
      "metadata": {
        "id": "70UWNT3IzMme"
      }
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "_7mz3kjQzK_S"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
