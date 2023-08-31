import logging
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackContext,
    AIORateLimiter
)
from database import Database
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MAX_LEN_RESPONSE = 3 # 1/2/3/-
PATH = 'models//model_GPT.pt'
# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

checkpoint = "Kirili4ik/ruDialoGpt3-medium-finetuned-telegram"
tokenizer =  AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

checkpoint = torch.load(PATH, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()



# Create a database instance
database = Database()

def get_length_param(text: str, tokenizer) -> str:
    tokens_count = len(tokenizer.encode(text))
    if tokens_count <= 15:
        len_param = '1'
    elif tokens_count <= 50:
        len_param = '2'
    elif tokens_count <= 256:
        len_param = '3'
    else:
        len_param = '-'
    return len_param

# Define command handlers
async def start(update: Update, context: CallbackContext) -> None:
    """Send a welcome message when the command /start is issued"""
    await register_user(update, context)
    await update.message.reply_text('🤖 Привет! Напиши мне сообщение. А если хочешь удалить контекст отправь /deletecontext')

async def echo(update: Update, context: CallbackContext) -> None:
    """Echo the user message and store it in the database"""
    message = update.message.text
    user_id = update.message.from_user.id
    response = generate_response(user_id, message)
    store_request(message, response)
    await update.message.reply_text(response)

async def delete_conv_context_in_db(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id   
    database.delete_conv_context(user_id)
    await update.message.reply_text('☠️ Контекст удален. Обычно бот сохраняет информацию о диалге при формировании ответа.')
                        

def prepare_conv_context_from_db(conv_context: str, message: str, next_len = MAX_LEN_RESPONSE):
    if conv_context == '':
        chat_history_ids = torch.zeros((1, 0), dtype=torch.int)
    else:
        chat_history_ids = [int(id) for id in conv_context.split(",")]  # Convert string to list
        chat_history_ids = torch.tensor(chat_history_ids, dtype=torch.int32)
    
    new_user_input_ids_1 = tokenizer.encode(f"|0|{get_length_param(message, tokenizer)}|" \
                                              + message + tokenizer.eos_token, return_tensors="pt")
    chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids_1], dim=-1)
    new_user_input_ids_2 = tokenizer.encode(f"|1|{next_len}|", return_tensors="pt")
    chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids_2], dim=-1)
    
    return chat_history_ids

def store_conv_context(chat_history_ids): # torch tensor
    conv_context = ','.join(str(id) for id in chat_history_ids)
    database.insert_conv_context(conv_context)


def generate_response(user_id: int, message: str) -> str:
    """Generate a response using the language model"""
    conv_context = database.select_conv_context(user_id)
    chat_history_ids = prepare_conv_context_from_db(conv_context, message)
    input_len = chat_history_ids.shape[-1]
    chat_history_ids = model.generate(
        chat_history_ids,
        num_return_sequences=1,                    
        max_length=512,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature = 1.6,                        
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    response = tokenizer.decode(chat_history_ids[:, input_len:][0], skip_special_tokens=True)
    store_conv_context(chat_history_ids)
    return response

def store_request(message: str, response: str) -> None:
    """Store the user request and bot response in the database"""
    database.insert_request(message, response)

async def register_user(update: Update, context: CallbackContext) -> None:
    """Register a new user and store the registration information in the database"""
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    first_name = update.message.from_user.first_name
    last_name = update.message.from_user.last_name

    # Check if the user already exists in the database
    if database.check_user_exists(user_id):
        return

    # Store the registration information in the database
    database.insert_user(user_id, username, first_name, last_name)

def main() -> None:
    """Start the bot"""
    # Create the Updater and pass it your bot's token
    TOKEN = "" # your token
    #updater = updater(token=TOKEN, use_context=True)
    app = ApplicationBuilder().token(TOKEN).concurrent_updates(True).build()

    # Get the dispatcher to register handlers
    #dispatcher = updater.dispatcher

    # Add command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), echo))
    app.add_handler(CommandHandler("deletecontext", delete_conv_context_in_db))

    # Start the bot
    app.run_polling()

if __name__ == '__main__':
    main()