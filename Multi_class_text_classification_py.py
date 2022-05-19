{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/SubhamZap/LSTM-Text-Classification/blob/main/Multi_class_text_classification_py.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1L39vQBwfEni",
        "outputId": "1dddfc04-356e-4aef-d407-3f5309b09a54"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Unzipping corpora/stopwords.zip.\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "True"
            ]
          },
          "metadata": {},
          "execution_count": 1
        }
      ],
      "source": [
        "import nltk\n",
        "nltk.download('stopwords')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "QaK8KXNOcDZM"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import re\n",
        "from nltk.corpus import stopwords\n",
        "from nltk import word_tokenize\n",
        "STOPWORDS = set(stopwords.words('english'))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0MsLrTmJOw1N"
      },
      "source": [
        "# **Tweets**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "T0Umbp8EcdEb",
        "outputId": "3196edec-fa64-41a1-f288-c3dbcd1fb3e3"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Unnamed: 0 Hastag                                              Tweet\n",
              "0           0  #love  RT I wonder why you chose me among these flowe...\n",
              "1           1  #love  Is any man called being circumcised let him no...\n",
              "2           2  #love  RT mx wwmx love s Shape Of Love has surpassed ...\n",
              "3           3  #love  Thank you foster carers amp fostering communit...\n",
              "4           4  #love  RT 22 NFTGiveaway freenft Love Manifest Giveaw..."
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-d929a493-0174-46a4-aae2-b4c10dd87fec\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Unnamed: 0</th>\n",
              "      <th>Hastag</th>\n",
              "      <th>Tweet</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>#love</td>\n",
              "      <td>RT I wonder why you chose me among these flowe...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>#love</td>\n",
              "      <td>Is any man called being circumcised let him no...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>2</td>\n",
              "      <td>#love</td>\n",
              "      <td>RT mx wwmx love s Shape Of Love has surpassed ...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>3</td>\n",
              "      <td>#love</td>\n",
              "      <td>Thank you foster carers amp fostering communit...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>4</td>\n",
              "      <td>#love</td>\n",
              "      <td>RT 22 NFTGiveaway freenft Love Manifest Giveaw...</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-d929a493-0174-46a4-aae2-b4c10dd87fec')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-d929a493-0174-46a4-aae2-b4c10dd87fec button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-d929a493-0174-46a4-aae2-b4c10dd87fec');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ],
      "source": [
        "df = pd.read_csv('tweets.csv')\n",
        "df.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Kn7iGIZK7s-V",
        "outputId": "3224962a-47b5-4fb4-c730-4a70c566b594"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Index(['Unnamed: 0', 'Hastag', 'Tweet'], dtype='object')"
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ],
      "source": [
        "df.columns"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2FUK_KXYdLLa",
        "outputId": "3a36d1dd-212d-41f9-b0b8-b600ad81a115"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(43988, 3)"
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ],
      "source": [
        "df.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fWkzSS6I5HiE"
      },
      "outputs": [],
      "source": [
        "df.rename(columns = {'Hastag': 'category', 'Tweet': 'tweet'}, inplace = True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vREYxLCA5w14",
        "outputId": "9a8d575d-dc40-4e86-aefd-0af396cc0666"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/pandas/core/indexing.py:1732: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  self._setitem_single_block(indexer, value, name)\n"
          ]
        }
      ],
      "source": [
        "import re\n",
        "\n",
        "for i in range(len(df)):\n",
        "  df['category'].iloc[i] = ''.join(re.sub(\"(@[A-Za-z0-9]+)|([^0-9A-Za-z \\t])|(\\w+:\\/\\/\\S+)\",\"\",df['category'].iloc[i]))\n",
        "  df['tweet'].iloc[i] = ''.join(re.sub(\"(@[A-Za-z0-9]+)|([^0-9A-Za-z \\t])|(\\w+:\\/\\/\\S+)\",\"\",df['tweet'].iloc[i]))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "zL0N1pfM7pxq"
      },
      "outputs": [],
      "source": [
        "df.drop(['Unnamed: 0'], axis = 1, inplace = True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sND3e4XV5aMc",
        "outputId": "8d9b0834-88e1-49c4-e1c1-35b5c5b9ec46"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array(['love', 'confession', 'job', 'startup', 'Politics', 'Movie',\n",
              "       'Game', 'Media', 'News', 'Technology', 'Finance'], dtype=object)"
            ]
          },
          "metadata": {},
          "execution_count": 11
        }
      ],
      "source": [
        "df.category.unique()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sym1-DoOc8rX",
        "outputId": "3e2427b4-629c-4ed6-be3d-fe778853e31f"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "love          5000\n",
              "job           5000\n",
              "startup       5000\n",
              "Politics      5000\n",
              "Movie         5000\n",
              "Finance       5000\n",
              "Media         3500\n",
              "News          3500\n",
              "Technology    3500\n",
              "Game          3145\n",
              "confession     343\n",
              "Name: category, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 12
        }
      ],
      "source": [
        "df.category.value_counts()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CTZ8E65ieGja"
      },
      "source": [
        "# *Text Cleaning*"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "kblazd_ud2n6"
      },
      "outputs": [],
      "source": [
        "space = re.compile('[/(){}\\[\\]\\|@,;]')\n",
        "symbols= re.compile('[^0-9a-z #+_]')\n",
        "STOPWORDS = set(stopwords.words('english'))\n",
        "\n",
        "def clean_text(text):\n",
        "    text = text.lower() # lowercase text\n",
        "    text = space.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.\n",
        "    text = symbols.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. \n",
        "    text = text.replace('x', '')\n",
        "#    text = re.sub(r'\\W+', '', text)\n",
        "    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text\n",
        "    return text"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bBBdPI2WfWE6"
      },
      "outputs": [],
      "source": [
        "df['cleaned_tweet'] = df.tweet.apply(lambda x : clean_text(x))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "4XVroQ2sfhzZ",
        "outputId": "acbe55f8-6fe8-4cf8-9f67-b996d4f3d69e"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "  category                                              tweet  \\\n",
              "0     love  RT I wonder why you chose me among these flowe...   \n",
              "1     love  Is any man called being circumcised let him no...   \n",
              "2     love  RT mx wwmx love s Shape Of Love has surpassed ...   \n",
              "3     love  Thank you foster carers amp fostering communit...   \n",
              "4     love  RT 22 NFTGiveaway freenft Love Manifest Giveaw...   \n",
              "\n",
              "                                       cleaned_tweet  \n",
              "0  rt wonder chose among flowers beautiful tree h...  \n",
              "1  man called circumcised let become uncircumcise...  \n",
              "2  rt wwm love shape love surpassed 345k copies s...  \n",
              "3  thank foster carers amp fostering communities ...  \n",
              "4  rt 22 nftgiveaway freenft love manifest giveaw...  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-88e71d9f-b378-4c45-82db-717e3966a286\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>category</th>\n",
              "      <th>tweet</th>\n",
              "      <th>cleaned_tweet</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>love</td>\n",
              "      <td>RT I wonder why you chose me among these flowe...</td>\n",
              "      <td>rt wonder chose among flowers beautiful tree h...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>love</td>\n",
              "      <td>Is any man called being circumcised let him no...</td>\n",
              "      <td>man called circumcised let become uncircumcise...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>love</td>\n",
              "      <td>RT mx wwmx love s Shape Of Love has surpassed ...</td>\n",
              "      <td>rt wwm love shape love surpassed 345k copies s...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>love</td>\n",
              "      <td>Thank you foster carers amp fostering communit...</td>\n",
              "      <td>thank foster carers amp fostering communities ...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>love</td>\n",
              "      <td>RT 22 NFTGiveaway freenft Love Manifest Giveaw...</td>\n",
              "      <td>rt 22 nftgiveaway freenft love manifest giveaw...</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-88e71d9f-b378-4c45-82db-717e3966a286')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-88e71d9f-b378-4c45-82db-717e3966a286 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-88e71d9f-b378-4c45-82db-717e3966a286');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ],
      "source": [
        "df.head()"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df[df['category'] == 'Movie']"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 424
        },
        "id": "cf3nEBOqwSwU",
        "outputId": "2bd97c2d-07ac-4d05-8e5f-99adf7a8ad41"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "      category                                              tweet  \\\n",
              "20343    Movie  RT acc who is favourite bollywood actress Retw...   \n",
              "20344    Movie   watching Battle Beyond The Sun space scifi movie   \n",
              "20345    Movie  RT I was lucky enough to get Annabelle what ab...   \n",
              "20346    Movie  RT GaryK Spectacular movie poster artwork by R...   \n",
              "20347    Movie  Love discussing every aspect of a movie or jus...   \n",
              "...        ...                                                ...   \n",
              "25338    Movie  Best UNIVERSAL VIDEO CASTING PLAYER CAST ALL V...   \n",
              "25339    Movie  horror kkndidesapenari kkndesapenari movies mo...   \n",
              "25340    Movie  How to Cast Play Videos from PC to Philips TV ...   \n",
              "25341    Movie  NowPlaying 1st Movie of the day Rewatching The...   \n",
              "25342    Movie  How to PLAY Videos from PC to Panasonic TV Bes...   \n",
              "\n",
              "                                           cleaned_tweet  \n",
              "20343  rt acc favourite bollywood actress retweet kia...  \n",
              "20344       watching battle beyond sun space scifi movie  \n",
              "20345  rt lucky enough get annabelle everyone else ch...  \n",
              "20346  rt garyk spectacular movie poster artwork reyn...  \n",
              "20347  love discussing every aspect movie want hear l...  \n",
              "...                                                  ...  \n",
              "25338  best universal video casting player cast video...  \n",
              "25339  horror kkndidesapenari kkndesapenari movies mo...  \n",
              "25340  cast play videos pc philips tv best universal ...  \n",
              "25341  nowplaying 1st movie day rewatching thefunhous...  \n",
              "25342  play videos pc panasonic tv best universal vid...  \n",
              "\n",
              "[5000 rows x 3 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-30bcfa1e-a3b3-4413-ad3e-1c77550287ec\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>category</th>\n",
              "      <th>tweet</th>\n",
              "      <th>cleaned_tweet</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>20343</th>\n",
              "      <td>Movie</td>\n",
              "      <td>RT acc who is favourite bollywood actress Retw...</td>\n",
              "      <td>rt acc favourite bollywood actress retweet kia...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>20344</th>\n",
              "      <td>Movie</td>\n",
              "      <td>watching Battle Beyond The Sun space scifi movie</td>\n",
              "      <td>watching battle beyond sun space scifi movie</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>20345</th>\n",
              "      <td>Movie</td>\n",
              "      <td>RT I was lucky enough to get Annabelle what ab...</td>\n",
              "      <td>rt lucky enough get annabelle everyone else ch...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>20346</th>\n",
              "      <td>Movie</td>\n",
              "      <td>RT GaryK Spectacular movie poster artwork by R...</td>\n",
              "      <td>rt garyk spectacular movie poster artwork reyn...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>20347</th>\n",
              "      <td>Movie</td>\n",
              "      <td>Love discussing every aspect of a movie or jus...</td>\n",
              "      <td>love discussing every aspect movie want hear l...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25338</th>\n",
              "      <td>Movie</td>\n",
              "      <td>Best UNIVERSAL VIDEO CASTING PLAYER CAST ALL V...</td>\n",
              "      <td>best universal video casting player cast video...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25339</th>\n",
              "      <td>Movie</td>\n",
              "      <td>horror kkndidesapenari kkndesapenari movies mo...</td>\n",
              "      <td>horror kkndidesapenari kkndesapenari movies mo...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25340</th>\n",
              "      <td>Movie</td>\n",
              "      <td>How to Cast Play Videos from PC to Philips TV ...</td>\n",
              "      <td>cast play videos pc philips tv best universal ...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25341</th>\n",
              "      <td>Movie</td>\n",
              "      <td>NowPlaying 1st Movie of the day Rewatching The...</td>\n",
              "      <td>nowplaying 1st movie day rewatching thefunhous...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25342</th>\n",
              "      <td>Movie</td>\n",
              "      <td>How to PLAY Videos from PC to Panasonic TV Bes...</td>\n",
              "      <td>play videos pc panasonic tv best universal vid...</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5000 rows × 3 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-30bcfa1e-a3b3-4413-ad3e-1c77550287ec')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-30bcfa1e-a3b3-4413-ad3e-1c77550287ec button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-30bcfa1e-a3b3-4413-ad3e-1c77550287ec');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "osNjUXPrqUkY"
      },
      "outputs": [],
      "source": [
        "g = []\n",
        "for i in df['cleaned_tweet']:\n",
        "    g.append(i)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jaxhDt_pqWqs",
        "outputId": "e9662273-d07e-4a88-f1d2-7fddf3e9da12"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Maximum sequence length in the list of sentences: 266\n"
          ]
        }
      ],
      "source": [
        "maxl = max([len(s) for s in g])\n",
        "print ('Maximum sequence length in the list of sentences:', maxl)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MQT5m9-Jqm5i",
        "outputId": "cf995ded-59e2-47e1-f1ef-3a03b425075a"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 53458 unique tokens.\n"
          ]
        }
      ],
      "source": [
        "from keras.preprocessing.text import Tokenizer\n",
        "\n",
        "tokenizer = Tokenizer(num_words=50000, filters='!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~', lower=True)\n",
        "tokenizer.fit_on_texts(df['cleaned_tweet'].values)\n",
        "word_index = tokenizer.word_index\n",
        "print('Found %s unique tokens.' % len(word_index))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6Oqb94acqpsA"
      },
      "outputs": [],
      "source": [
        "from keras.preprocessing.sequence import pad_sequences\n",
        "\n",
        "X = tokenizer.texts_to_sequences(df['cleaned_tweet'].values)\n",
        "X = pad_sequences(X, maxlen=1000)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "tvKZiqDwqtYm"
      },
      "outputs": [],
      "source": [
        "y = pd.get_dummies(df['category'],columns=df[\"category\"]).values"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "x-zLa_8WjMPf"
      },
      "outputs": [],
      "source": [
        "from sklearn.model_selection import train_test_split"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "OmzrlWTHjo-v"
      },
      "outputs": [],
      "source": [
        "x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-BsGL-bqj8O_",
        "outputId": "9ffd338d-193b-4173-b178-08d89c0c96c5"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "shape of x_train: (35190, 1000)\n",
            "shape of x_test: (8798, 1000)\n",
            "shape of y_train: (35190, 11)\n",
            "shape of y_test: (8798, 11)\n"
          ]
        }
      ],
      "source": [
        "print('shape of x_train:', x_train.shape)\n",
        "print('shape of x_test:', x_test.shape)\n",
        "print('shape of y_train:', y_train.shape)\n",
        "print('shape of y_test:', y_test.shape)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iLrHkXlFklgn",
        "outputId": "01c19237-a18d-47d4-e749-c5559e323551"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[    0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "     0     0     0     0     0     0     0     0     0     0     0     0\n",
            "    83  1486   257  8624  6583  5107   303    83  1257  5365 17970   716\n",
            "   575   990     6  3507]\n"
          ]
        }
      ],
      "source": [
        "print(x_train[4])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "E5lU0ZfVmnUa"
      },
      "outputs": [],
      "source": [
        "from keras.models import Sequential\n",
        "from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D\n",
        "from sklearn.model_selection import train_test_split\n",
        "from keras.utils.np_utils import to_categorical\n",
        "from keras.callbacks import EarlyStopping\n",
        "from keras.layers import Dropout"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "eA2D3NsLljAv"
      },
      "outputs": [],
      "source": [
        "model=Sequential()\n",
        "model.add(Embedding(50000,100,input_length=1000))\n",
        "model.add(SpatialDropout1D(0.2))\n",
        "model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))\n",
        "model.add(Dense(11, activation='softmax'))\n",
        "model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fwyq3SuPmpxF",
        "outputId": "5a9ee1e0-55e0-4855-d0cf-5e67bf621dee"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"sequential\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " embedding (Embedding)       (None, 1000, 100)         5000000   \n",
            "                                                                 \n",
            " spatial_dropout1d (SpatialD  (None, 1000, 100)        0         \n",
            " ropout1D)                                                       \n",
            "                                                                 \n",
            " lstm (LSTM)                 (None, 100)               80400     \n",
            "                                                                 \n",
            " dense (Dense)               (None, 11)                1111      \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 5,081,511\n",
            "Trainable params: 5,081,511\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ]
        }
      ],
      "source": [
        "model.summary()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 29,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fotXd_j-mtv-",
        "outputId": "b78422bd-ea65-438a-b42f-d3a4a82cd117"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "495/495 [==============================] - 1391s 3s/step - loss: 0.7308 - accuracy: 0.7849 - val_loss: 0.2716 - val_accuracy: 0.9167\n",
            "Epoch 2/10\n",
            "495/495 [==============================] - 1414s 3s/step - loss: 0.1739 - accuracy: 0.9469 - val_loss: 0.2290 - val_accuracy: 0.9295\n",
            "Epoch 3/10\n",
            "495/495 [==============================] - 1466s 3s/step - loss: 0.0999 - accuracy: 0.9656 - val_loss: 0.2310 - val_accuracy: 0.9253\n",
            "Epoch 4/10\n",
            "495/495 [==============================] - 1457s 3s/step - loss: 0.0777 - accuracy: 0.9705 - val_loss: 0.2443 - val_accuracy: 0.9270\n",
            "Epoch 5/10\n",
            "495/495 [==============================] - 1493s 3s/step - loss: 0.0670 - accuracy: 0.9721 - val_loss: 0.2513 - val_accuracy: 0.9255\n"
          ]
        }
      ],
      "source": [
        "history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 30,
      "metadata": {
        "id": "MrvlMowjm26u",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "82bd630a-b680-448d-f086-b5ca670c00ad"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "275/275 [==============================] - 67s 243ms/step - loss: 0.2574 - accuracy: 0.9286\n",
            "Test set\n",
            "  Loss: 0.257\n",
            "  Accuracy: 0.929\n"
          ]
        }
      ],
      "source": [
        "accr = model.evaluate(x_test,y_test)\n",
        "print('Test set\\n  Loss: {:0.3f}\\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pickle\n",
        "pickle_out = open(\"classifier.pkl\", \"wb\")\n",
        "pickle.dump(model, pickle_out)\n",
        "pickle_out.close()"
      ],
      "metadata": {
        "id": "TKGlw7hTvYc0",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "428f5ca6-3ed4-4c2d-9bfe-d98b65b33b4d"
      },
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "INFO:tensorflow:Assets written to: ram://67a0d1cf-4f36-4808-b977-7e1c721fe338/assets\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:absl:<keras.layers.recurrent.LSTMCell object at 0x7f1ae1b57250> has the same name 'LSTMCell' as a built-in Keras object. Consider renaming <class 'keras.layers.recurrent.LSTMCell'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "jDMsQO4zn9Df"
      },
      "outputs": [],
      "source": [
        "plt.title('Loss')\n",
        "plt.plot(history.history['loss'], label='train')\n",
        "plt.plot(history.history['val_loss'], label='test')\n",
        "plt.legend()\n",
        "plt.show();"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 32,
      "metadata": {
        "id": "YmfdDUp9oEVI",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b5b9e2f4-3968-4d14-acd4-bae2f473f893"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[8.4083650e-04 4.2953838e-05 9.9145365e-01 3.1022204e-04 5.5653416e-03\n",
            "  3.7634867e-04 5.9361832e-04 2.6539114e-05 1.2765090e-04 6.4986397e-04\n",
            "  1.2983675e-05]] job\n"
          ]
        }
      ],
      "source": [
        "new_complaint = ['Tata Sons Chairman Emeritus Ratan Tata was spotted visiting Taj Hotel in his Tata Nano without any bodyguard accompanying him, a video shared by Viral Bhayani showed. Bhayani said that Ratan Tata was escorted by the hotel staff. Several people praised him on social media, with a social media user saying, \"Exemplary simplicity personified.\"']\n",
        "seq = tokenizer.texts_to_sequences(new_complaint)\n",
        "padded = pad_sequences(seq, maxlen=1000)\n",
        "pred = model.predict(padded)\n",
        "labels = ['love', 'confession', 'job', 'startup', 'Politics', 'Movie', 'Game', 'Media', 'News', 'Technology', 'Finance']\n",
        "print(pred, labels[np.argmax(pred)])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 33,
      "metadata": {
        "id": "y768RUjKBPyJ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "9b8e6b7d-01c2-4bb1-e043-7ca79dcf055b"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0.31579995 0.0054747  0.05892811 0.00465872 0.11427016 0.34531143\n",
            "  0.00210473 0.05322683 0.09446543 0.00113584 0.00462421]] Movie\n"
          ]
        }
      ],
      "source": [
        "new_complaint =['Congress leader Sachin Pilot targeted the BJP-led central government over rising inflation and unemployment in the country. \"Inflation and unemployment are making new records day by day under the directionless and failed governance of the central government,\" he tweeted. \"With...wholesale inflation registering 15.08% in...April, inflation...has reached the highest level of 24 years,\" he added.']\n",
        "seq = tokenizer.texts_to_sequences(new_complaint)\n",
        "padded = pad_sequences(seq, maxlen=1000)\n",
        "pred = model.predict(padded)\n",
        "labels = ['love', 'confession', 'job', 'startup', 'Politics', 'Movie', 'Game', 'Media', 'News', 'Technology', 'Finance']\n",
        "print(pred, labels[np.argmax(pred)])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 34,
      "metadata": {
        "id": "ZMnNnD5tBcH_",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "6e5ba74b-9d0c-4fcc-eb2f-a1396cbb7b5d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[5.3702457e-05 1.4112274e-03 1.6165088e-03 4.0912300e-02 4.5299393e-04\n",
            "  1.3883020e-03 5.5483612e-04 8.7220659e-03 6.9692492e-04 9.4361734e-01\n",
            "  5.7383411e-04]] Technology\n"
          ]
        }
      ],
      "source": [
        "new_complaint = ['In a recent interview, when actress Kareena Kapoor was asked which iconic look from her films shed like to recreate, Kareena said, \"Since Poo from Kabhi Khushi Kabhie Gham is one of my most iconic characters, I would love to recreate that look of mine.\" Meanwhile, on the work front, Kareena has begun shooting for Sujoy Ghoshs next.']\n",
        "seq = tokenizer.texts_to_sequences(new_complaint)\n",
        "padded = pad_sequences(seq, maxlen=1000)\n",
        "pred = model.predict(padded)\n",
        "labels = ['love', 'confession', 'job', 'startup', 'Politics', 'Movie', 'Game', 'Media', 'News', 'Technology', 'Finance']\n",
        "print(pred, labels[np.argmax(pred)])"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "new_complaint =['Ruchi Soya on Wednesday said itll be acquiring Patanjali Ayurveds food retail business for an amount of consideration of ₹690 crore.']\n",
        "seq = tokenizer.texts_to_sequences(new_complaint)\n",
        "padded = pad_sequences(seq, maxlen=1000)\n",
        "pred = model.predict(padded)\n",
        "labels = ['love', 'confession', 'job', 'startup', 'Politics', 'Movie', 'Game', 'Media', 'News', 'Technology', 'Finance']\n",
        "print(pred, labels[np.argmax(pred)])"
      ],
      "metadata": {
        "id": "-efsjt6nwDPW",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "89d6fe48-7e6e-4019-fd88-0a744a07fed1"
      },
      "execution_count": 35,
      "outputs": [
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "[[0.3082348  0.0173871  0.01491851 0.01658555 0.0189406  0.01845297\n",
            "  0.3362218  0.00131564 0.10115436 0.00650285 0.16028585]] Game\n"
          ]
        }
      ]
    }
  ],
  "metadata": {
    "colab": {
      "collapsed_sections": [],
      "name": "Multi class text classification.py",
      "provenance": [],
      "authorship_tag": "ABX9TyOduX+5gagIQFlB54E4+Fgl",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}