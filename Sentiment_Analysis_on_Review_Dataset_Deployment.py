{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Sentiment_Analysis_on_Review_Dataset_Deployment.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyOP5qcMTdRbBKIrDYgvx7V+",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/surajpai50612/Mini-Project/blob/master/Sentiment_Analysis_on_Review_Dataset_Deployment.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "RlPbhuZNz38x",
        "outputId": "26a3dc39-7307-4cfd-e535-a8b57b058ac4",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "%%writefile app1.py\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "\n",
        "df = pd.read_csv('sentiment_review.csv')\n",
        "x = df.iloc[:,0].values # Message column as input\n",
        "y = df.iloc[:,1].values # Label column as output\n",
        "st.title(\"Spam Ham Message Classifier\")\n",
        "st.subheader('TFIFD Vectorizer')\n",
        "st.write('This project is based on Naive Bayes Classifier')\n",
        "\n",
        "text_model = Pipeline([('tfidf',TfidfVectorizer()),('model',MultinomialNB())])\n",
        "text_model.fit(x,y)\n",
        "message = st.text_area(\"Enter Text\",\"Type Here ..\")\n",
        "op = text_model.predict([message])\n",
        "if st.button(\"Predict\"):\n",
        "  st.title(op)\n"
      ],
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Overwriting app1.py\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "qPgV9zyz0fw9",
        "outputId": "1a9cd7f4-ef1f-434a-b6e6-e8dd6c4f95cb",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        }
      },
      "source": [
        "!pip install streamlit"
      ],
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Collecting streamlit\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/6c/0d/bb0bd245beeeee38a19286718069ffeef1c651106cd4bbebae9091e4fc1f/streamlit-0.67.1-py2.py3-none-any.whl (7.2MB)\n",
            "\u001b[K     |████████████████████████████████| 7.2MB 2.5MB/s \n",
            "\u001b[?25hRequirement already satisfied: pandas>=0.21.0 in /usr/local/lib/python3.6/dist-packages (from streamlit) (1.1.2)\n",
            "Requirement already satisfied: toml in /usr/local/lib/python3.6/dist-packages (from streamlit) (0.10.1)\n",
            "Requirement already satisfied: pyarrow in /usr/local/lib/python3.6/dist-packages (from streamlit) (0.14.1)\n",
            "Requirement already satisfied: click>=7.0 in /usr/local/lib/python3.6/dist-packages (from streamlit) (7.1.2)\n",
            "Collecting botocore>=1.13.44\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/cb/3f/e55f26983b7315265066f1cdc614e8d7e2c6a7a798ac6be88f3fdff90533/botocore-1.18.10-py2.py3-none-any.whl (6.7MB)\n",
            "\u001b[K     |████████████████████████████████| 6.7MB 39.5MB/s \n",
            "\u001b[?25hRequirement already satisfied: cachetools>=4.0 in /usr/local/lib/python3.6/dist-packages (from streamlit) (4.1.1)\n",
            "Requirement already satisfied: altair>=3.2.0 in /usr/local/lib/python3.6/dist-packages (from streamlit) (4.1.0)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.6/dist-packages (from streamlit) (20.4)\n",
            "Collecting enum-compat\n",
            "  Downloading https://files.pythonhosted.org/packages/55/ae/467bc4509246283bb59746e21a1a2f5a8aecbef56b1fa6eaca78cd438c8b/enum_compat-0.0.3-py3-none-any.whl\n",
            "Requirement already satisfied: astor in /usr/local/lib/python3.6/dist-packages (from streamlit) (0.8.1)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from streamlit) (2.23.0)\n",
            "Collecting validators\n",
            "  Downloading https://files.pythonhosted.org/packages/41/4a/3360ff3cf2b4a1b9721ac1fbff5f84663f41047d9874b3aa1ac82e862c44/validators-0.18.1-py3-none-any.whl\n",
            "Requirement already satisfied: protobuf>=3.6.0 in /usr/local/lib/python3.6/dist-packages (from streamlit) (3.12.4)\n",
            "Collecting watchdog\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/0e/06/121302598a4fc01aca942d937f4a2c33430b7181137b35758913a8db10ad/watchdog-0.10.3.tar.gz (94kB)\n",
            "\u001b[K     |████████████████████████████████| 102kB 7.8MB/s \n",
            "\u001b[?25hRequirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from streamlit) (1.18.5)\n",
            "Requirement already satisfied: python-dateutil in /usr/local/lib/python3.6/dist-packages (from streamlit) (2.8.1)\n",
            "Collecting blinker\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/1b/51/e2a9f3b757eb802f61dc1f2b09c8c99f6eb01cf06416c0671253536517b6/blinker-1.4.tar.gz (111kB)\n",
            "\u001b[K     |████████████████████████████████| 112kB 32.5MB/s \n",
            "\u001b[?25hRequirement already satisfied: tornado>=5.0 in /usr/local/lib/python3.6/dist-packages (from streamlit) (5.1.1)\n",
            "Collecting boto3\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/33/73/e90df0e0a5c55291273697d8eaa33f920b10590b5f632e971bb68ec55988/boto3-1.15.10-py2.py3-none-any.whl (129kB)\n",
            "\u001b[K     |████████████████████████████████| 133kB 47.1MB/s \n",
            "\u001b[?25hRequirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.6/dist-packages (from streamlit) (7.0.0)\n",
            "Requirement already satisfied: tzlocal in /usr/local/lib/python3.6/dist-packages (from streamlit) (1.5.1)\n",
            "Collecting pydeck>=0.1.dev5\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/51/1e/296f4108bf357e684617a776ecaf06ee93b43e30c35996dfac1aa985aa6c/pydeck-0.5.0b1-py2.py3-none-any.whl (4.4MB)\n",
            "\u001b[K     |████████████████████████████████| 4.4MB 40.3MB/s \n",
            "\u001b[?25hCollecting base58\n",
            "  Downloading https://files.pythonhosted.org/packages/3c/03/58572025c77b9e6027155b272a1b96298e711cd4f95c24967f7137ab0c4b/base58-2.0.1-py3-none-any.whl\n",
            "Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas>=0.21.0->streamlit) (2018.9)\n",
            "Requirement already satisfied: six>=1.0.0 in /usr/local/lib/python3.6/dist-packages (from pyarrow->streamlit) (1.15.0)\n",
            "Requirement already satisfied: urllib3<1.26,>=1.20; python_version != \"3.4\" in /usr/local/lib/python3.6/dist-packages (from botocore>=1.13.44->streamlit) (1.24.3)\n",
            "Collecting jmespath<1.0.0,>=0.7.1\n",
            "  Downloading https://files.pythonhosted.org/packages/07/cb/5f001272b6faeb23c1c9e0acc04d48eaaf5c862c17709d20e3469c6e0139/jmespath-0.10.0-py2.py3-none-any.whl\n",
            "Requirement already satisfied: jsonschema in /usr/local/lib/python3.6/dist-packages (from altair>=3.2.0->streamlit) (2.6.0)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.6/dist-packages (from altair>=3.2.0->streamlit) (0.11.1)\n",
            "Requirement already satisfied: entrypoints in /usr/local/lib/python3.6/dist-packages (from altair>=3.2.0->streamlit) (0.3)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.6/dist-packages (from altair>=3.2.0->streamlit) (2.11.2)\n",
            "Requirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.6/dist-packages (from packaging->streamlit) (2.4.7)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->streamlit) (2.10)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests->streamlit) (2020.6.20)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->streamlit) (3.0.4)\n",
            "Requirement already satisfied: decorator>=3.4.0 in /usr/local/lib/python3.6/dist-packages (from validators->streamlit) (4.4.2)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.6/dist-packages (from protobuf>=3.6.0->streamlit) (50.3.0)\n",
            "Collecting pathtools>=0.1.1\n",
            "  Downloading https://files.pythonhosted.org/packages/e7/7f/470d6fcdf23f9f3518f6b0b76be9df16dcc8630ad409947f8be2eb0ed13a/pathtools-0.1.2.tar.gz\n",
            "Collecting s3transfer<0.4.0,>=0.3.0\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/69/79/e6afb3d8b0b4e96cefbdc690f741d7dd24547ff1f94240c997a26fa908d3/s3transfer-0.3.3-py2.py3-none-any.whl (69kB)\n",
            "\u001b[K     |████████████████████████████████| 71kB 6.7MB/s \n",
            "\u001b[?25hRequirement already satisfied: traitlets>=4.3.2 in /usr/local/lib/python3.6/dist-packages (from pydeck>=0.1.dev5->streamlit) (4.3.3)\n",
            "Requirement already satisfied: ipywidgets>=7.0.0 in /usr/local/lib/python3.6/dist-packages (from pydeck>=0.1.dev5->streamlit) (7.5.1)\n",
            "Collecting ipykernel>=5.1.2; python_version >= \"3.4\"\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/52/19/c2812690d8b340987eecd2cbc18549b1d130b94c5d97fcbe49f5f8710edf/ipykernel-5.3.4-py3-none-any.whl (120kB)\n",
            "\u001b[K     |████████████████████████████████| 122kB 43.2MB/s \n",
            "\u001b[?25hRequirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.6/dist-packages (from jinja2->altair>=3.2.0->streamlit) (1.1.1)\n",
            "Requirement already satisfied: ipython-genutils in /usr/local/lib/python3.6/dist-packages (from traitlets>=4.3.2->pydeck>=0.1.dev5->streamlit) (0.2.0)\n",
            "Requirement already satisfied: nbformat>=4.2.0 in /usr/local/lib/python3.6/dist-packages (from ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (5.0.7)\n",
            "Requirement already satisfied: ipython>=4.0.0; python_version >= \"3.3\" in /usr/local/lib/python3.6/dist-packages (from ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (5.5.0)\n",
            "Requirement already satisfied: widgetsnbextension~=3.5.0 in /usr/local/lib/python3.6/dist-packages (from ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (3.5.1)\n",
            "Requirement already satisfied: jupyter-client in /usr/local/lib/python3.6/dist-packages (from ipykernel>=5.1.2; python_version >= \"3.4\"->pydeck>=0.1.dev5->streamlit) (5.3.5)\n",
            "Requirement already satisfied: jupyter-core in /usr/local/lib/python3.6/dist-packages (from nbformat>=4.2.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (4.6.3)\n",
            "Requirement already satisfied: pickleshare in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0; python_version >= \"3.3\"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (0.7.5)\n",
            "Requirement already satisfied: pexpect; sys_platform != \"win32\" in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0; python_version >= \"3.3\"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (4.8.0)\n",
            "Requirement already satisfied: simplegeneric>0.8 in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0; python_version >= \"3.3\"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (0.8.1)\n",
            "Requirement already satisfied: prompt-toolkit<2.0.0,>=1.0.4 in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0; python_version >= \"3.3\"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (1.0.18)\n",
            "Requirement already satisfied: pygments in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0; python_version >= \"3.3\"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (2.6.1)\n",
            "Requirement already satisfied: notebook>=4.4.1 in /usr/local/lib/python3.6/dist-packages (from widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (5.3.1)\n",
            "Requirement already satisfied: pyzmq>=13 in /usr/local/lib/python3.6/dist-packages (from jupyter-client->ipykernel>=5.1.2; python_version >= \"3.4\"->pydeck>=0.1.dev5->streamlit) (19.0.2)\n",
            "Requirement already satisfied: ptyprocess>=0.5 in /usr/local/lib/python3.6/dist-packages (from pexpect; sys_platform != \"win32\"->ipython>=4.0.0; python_version >= \"3.3\"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (0.6.0)\n",
            "Requirement already satisfied: wcwidth in /usr/local/lib/python3.6/dist-packages (from prompt-toolkit<2.0.0,>=1.0.4->ipython>=4.0.0; python_version >= \"3.3\"->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (0.2.5)\n",
            "Requirement already satisfied: nbconvert in /usr/local/lib/python3.6/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (5.6.1)\n",
            "Requirement already satisfied: terminado>=0.8.1 in /usr/local/lib/python3.6/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (0.9.1)\n",
            "Requirement already satisfied: Send2Trash in /usr/local/lib/python3.6/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (1.5.0)\n",
            "Requirement already satisfied: mistune<2,>=0.8.1 in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (0.8.4)\n",
            "Requirement already satisfied: bleach in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (3.2.1)\n",
            "Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (1.4.2)\n",
            "Requirement already satisfied: testpath in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (0.4.4)\n",
            "Requirement already satisfied: defusedxml in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (0.6.0)\n",
            "Requirement already satisfied: webencodings in /usr/local/lib/python3.6/dist-packages (from bleach->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (0.5.1)\n",
            "Building wheels for collected packages: watchdog, blinker, pathtools\n",
            "  Building wheel for watchdog (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for watchdog: filename=watchdog-0.10.3-cp36-none-any.whl size=73873 sha256=d144b7e55d31a7d76aebcae2fba9a499107a27ab3508ad1f043d098306848588\n",
            "  Stored in directory: /root/.cache/pip/wheels/a8/1d/38/2c19bb311f67cc7b4d07a2ec5ea36ab1a0a0ea50db994a5bc7\n",
            "  Building wheel for blinker (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for blinker: filename=blinker-1.4-cp36-none-any.whl size=13450 sha256=22dd3355cc75d133c492b4c42c28d8e657c28e19c8c630d9e71d808c7daac7e6\n",
            "  Stored in directory: /root/.cache/pip/wheels/92/a0/00/8690a57883956a301d91cf4ec999cc0b258b01e3f548f86e89\n",
            "  Building wheel for pathtools (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for pathtools: filename=pathtools-0.1.2-cp36-none-any.whl size=8785 sha256=36aef8c56fae6695609034e8d97fce136140033a33cddd509f2bbbf09ab42bc3\n",
            "  Stored in directory: /root/.cache/pip/wheels/0b/04/79/c3b0c3a0266a3cb4376da31e5bfe8bba0c489246968a68e843\n",
            "Successfully built watchdog blinker pathtools\n",
            "\u001b[31mERROR: google-colab 1.0.0 has requirement ipykernel~=4.10, but you'll have ipykernel 5.3.4 which is incompatible.\u001b[0m\n",
            "Installing collected packages: jmespath, botocore, enum-compat, validators, pathtools, watchdog, blinker, s3transfer, boto3, ipykernel, pydeck, base58, streamlit\n",
            "  Found existing installation: ipykernel 4.10.1\n",
            "    Uninstalling ipykernel-4.10.1:\n",
            "      Successfully uninstalled ipykernel-4.10.1\n",
            "Successfully installed base58-2.0.1 blinker-1.4 boto3-1.15.10 botocore-1.18.10 enum-compat-0.0.3 ipykernel-5.3.4 jmespath-0.10.0 pathtools-0.1.2 pydeck-0.5.0b1 s3transfer-0.3.3 streamlit-0.67.1 validators-0.18.1 watchdog-0.10.3\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "application/vnd.colab-display-data+json": {
              "pip_warning": {
                "packages": [
                  "ipykernel"
                ]
              }
            }
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "TbfYokru0lFH",
        "outputId": "ac79abda-8bcf-4eaf-ebe5-9893037c7776",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 151
        }
      },
      "source": [
        "!streamlit run app1.py"
      ],
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.2:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://35.231.50.136:8501\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Stopping...\u001b[0m\n",
            "\u001b[34m  Stopping...\u001b[0m\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uRDuIU-l0xa0",
        "outputId": "1c748c10-0057-41e9-b7c5-7d0603e7abf9",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 67
        }
      },
      "source": [
        "!pip install pyngrok"
      ],
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: pyngrok in /usr/local/lib/python3.6/dist-packages (4.1.12)\n",
            "Requirement already satisfied: PyYAML in /usr/local/lib/python3.6/dist-packages (from pyngrok) (3.13)\n",
            "Requirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from pyngrok) (0.16.0)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "d_keu5E-02MO",
        "outputId": "df660eea-b333-4aa0-b937-167434630447",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        }
      },
      "source": [
        "from pyngrok import ngrok\n",
        "url = ngrok.connect(port='8501')\n",
        "url"
      ],
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            },
            "text/plain": [
              "'http://aa26159db223.ngrok.io'"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 20
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "U22-T3x82TyA"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}