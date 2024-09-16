from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

# Initialize FastAPI app
app = FastAPI()

# Configure Gemini API
genai.configure("YOU API-KEY")


# Define the negotiation instructions
NEGOTIATION_PROMPT = (
    "You are a negotiation expert negotiating the price of a product. "
    "The current price is 10,000 INR. You will not accept any offer below 7,000 INR. "
    "The user can accept, reject, or make a counteroffer. "
    "Consider the user's sentiment in your response. If the user is polite, you may offer a better deal. "
    "Start the conversation by providing the selling price."
    "Do not jump directly to the lowest selling price. Make minimum discounts. Do not deduct more than 1000 at a time."
    "Accept the user price if the user is polite and it is greater than the minimum."
    "If the user is rude then, reject his offer."
    "Strictly adhere to these rules"
)

class UserInput(BaseModel):
    message: str

# Initialize conversation history
history = []

@app.post("/negotiate")
async def negotiate(user_input: UserInput = None):
    global history
    if not history:
        # Start with the initial negotiation message
        history = [
            {"role": "model", "parts": NEGOTIATION_PROMPT}
        ]

        # Initialize the model
        model = genai.GenerativeModel("gemini-1.5-flash")
        chat = model.start_chat(history=history)

        # Get the initial response from the model
        response = chat.send_message(NEGOTIATION_PROMPT)

        # Update history with the initial response from the model
        history.append({"role": "model", "parts": response.text})

        return {"response": response.text}

    # Add user input to history
    history.append({"role": "user", "parts": user_input.message})

    # Generate model response
    model = genai.GenerativeModel("gemini-1.5-flash")
    chat = model.start_chat(history=history)
    response = chat.send_message(user_input.message)

    # Update history with model response
    history.append({"role": "model", "parts": response.text})
    return {"response": response.text}

