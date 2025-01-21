# supervisor.py

from utils_cpu_quantize import LLMHandler
from utils_cpu_quantize import MODEL_ID  # Add this import


### Insctructions changed on 221st Jan
# Fixed Income Classification Instructions
CLASSIFICATION_INSTRUCTIONS = """Instruction: You are a supervisor tasked with managing the chat messages with queries about Fixed Income instruments, an AI assistant.
Your task is to classify the incoming questions. 
Depending on your answer, question will be routed to the right team, so your task is crucial for our team. 

There are 5 possible question types: 

- QuoteRequest - questions related to price, bid, offer for a given Fixed Income security.
- BondInfo - questions related to information about a Fixed Income security.
- Negotiation – question related to price negotiation 
- Settlement – questions related to settlement of transactions
- TransactionInfo – information about a transaction done or to be done 
- TransactionPlan  - questions related to planning for transaction execution 
- General - general questions

Return in the output only one word (BondInfo, Negotiation, Settlement, TransactionInfo  or General)


Examples:
Chat Message: 8.3774% HDB Financial Apr 26 INE756I07ER5 Qtm: 1 Cr Offer please
Output:QuoteRequest
Chat Message: can you please share a brief note about AP state bonds?
Output:BondInfo
Chat Message: How is the track record of profitability?
Output:BondInfo


Here are the messages: 
please share details of Indostar 
need details on UP Power Corp, Stanchart Securities One Pager I Payouts I Best Yields I Quantum 
Hi All - ne specific worry on Spandana 
ne chance the NIC is available again 
Hi, As discussed @~Bonds Prudent We will do Monday T+0 
Hello Sir Can we do today T+1? At 12.05% 
AROHAN FINANCIAL SERVICES LIMITED CLASS C 12.85 NCD 25OT26 FVRS10 INE808K08061 Qtm: 4 L Any bids pls? 
12.40 annapurna finance 12 Apr 2029 10 L selling Any bid pls? 
Hi Sir, KRAZYBEE 10.30 NCD 12JU26 ISIN : INE07HK07783 20 L looking to buy Bidding at 12.80% 
Holding 12.75% offer Please let me know if you can match the bid 
Funds Received Pls transfer the bonds

Chat Message: """

def create_prompt(user_input: str) -> str:
    """Create the complete prompt by combining the instruction and user input."""
    return CLASSIFICATION_INSTRUCTIONS + user_input

def main():
    """Main function to run the chatbot."""
    print(f"Fixed Income Query Classification Bot using {MODEL_ID} (type 'exit' to quit)")
    print("-" * 50)
    
    # Initialize LLM
    llm = LLMHandler()
    try:
        llm.initialize_llm()
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        return
    
    while True:
        try:
            # Get user input
            user_input = input("\nEnter your query: ").strip()
            
            # Check for exit command
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            # Skip empty inputs
            if not user_input:
                continue
            
            # Create prompt and get response
            prompt = create_prompt(user_input)
            response = llm.get_llm_response(prompt)
            
            # Print response
            print(f"\nClassification: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nGENERAL")  # Fallback response for any unexpected errors
            continue

if __name__ == "__main__":
    main()