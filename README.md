# Mental Health Companion

An AI-powered mental health support chatbot that provides empathetic responses and guidance for mental wellness. Built during the IIC Quest Hackathon.

**[Live Demo](https://mental-health-companion-momocoders.streamlit.app/)**

## Features

- Conversational AI for mental health support
- Empathetic and helpful responses
- Content safety detection (cyberbullying filter)
- Clean, calming user interface
- Privacy-focused design

## Tech Stack

- **Backend**: Node.js, Express, Prisma ORM
- **ML Model**: Naive Bayes classifier for content moderation
- **Frontend**: Streamlit
- **Language**: Python, JavaScript
- **Libraries**: scikit-learn, NLTK, NumPy

## Project Structure

```
├── app.py                 # Streamlit web application
├── model.pkl              # Content moderation model
├── vectorizer.pkl         # TF-IDF vectorizer
├── requirements.txt       # Python dependencies
├── dataset.json           # Mental health Q&A dataset (1,200+ pairs)
├── cyber_bullying.ipynb   # Model training notebook
└── backend/               # Node.js API server
```

## Local Setup

```bash
# Clone the repository
git clone https://github.com/YuneshShrestha/mental-health-companion.git
cd mental-health-companion

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## How It Works

1. **User Input**: User shares their thoughts or concerns
2. **Content Moderation**: ML model filters harmful content
3. **Response Generation**: AI provides supportive, empathetic responses
4. **Continuous Support**: Ongoing conversation for mental wellness

## Contributors

- Team Momo.Coders - IIC Quest Hackathon 2024

## Disclaimer

This chatbot is not a substitute for professional mental health care. If you're experiencing a crisis, please contact a mental health professional or crisis hotline.

## License

MIT License
