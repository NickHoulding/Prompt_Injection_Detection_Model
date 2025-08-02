import tensorflow as tf
import numpy as np
import argparse
import ollama
import os
from lr_model import load_model

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prompt Injection Detection: Interactive Demo"
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='nn', 
        help='Select either "nn" for neural network or "lr" for logistic regression model.'
    )
    return parser.parse_args()

def main():
    print("=== Prompt Injection Detection Demo ===\n")

    args = parse_args()
    model_type = args.model

    if model_type not in ['nn', 'lr']:
        print("[✗] Invalid model type specified. Use 'nn' for neural network or 'lr' for logistic regression.")
        return

    try:
        if model_type == 'lr':
            print("[-] Loading logistic regression model...")
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'lr_model.pkl')
            
            if not os.path.exists(model_path):
                print(f"[✗] Model file not found: {model_path}")
                return

            model = load_model(model_path)

        elif model_type == 'nn':
            print("[-] Loading neural network model...")
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'nn_model.keras')
            model = tf.keras.models.load_model(model_path)

    except Exception as e:
        print(f"[✗] Error loading model: {e}")
        return

    print("[✓] Model loaded successfully!\n")    
    print("Model is ready! You can now test prompts for injection detection.")
    print("Type 'quit', 'exit', or 'q' to stop.\n")
    
    while True:
        try:
            user_input = input("Enter a prompt to analyze: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("[✓] Goodbye!")
                break
            
            if not user_input:
                print("[✗] Please enter a non-empty prompt.\n")
                continue

            response = ollama.embed(
                model="nomic-embed-text",
                input=user_input,
            )

            if model_type == 'nn':
                X_input = np.array(response['embeddings']).reshape(1, -1)
                prediction = model.predict(X_input, verbose=0)
                probability = prediction[0, 0]

                is_malicious = probability > 0.5
                confidence = probability if is_malicious else 1 - probability
                raw_prob_malicious = probability
            elif model_type == 'lr':
                X_input = np.array(response['embeddings']).reshape(-1, 1)
                prediction = model.predict(X_input)
                probability = model.predict_proba(X_input)

                is_malicious = prediction[0, 0] == 1
                raw_prob_malicious = probability[0, 0]
                confidence = raw_prob_malicious if is_malicious else 1 - raw_prob_malicious
            
            print(f"\n--- Results ---")
            print(f"Classification: {'[!] MALICIOUS' if is_malicious else '[✓] LEGITIMATE'}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Raw probability (malicious): {raw_prob_malicious:.4f}")
            print("-" * 40 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error processing input: {e}\n")

if __name__ == "__main__":
    main()