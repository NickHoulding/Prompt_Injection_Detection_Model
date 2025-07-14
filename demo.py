import numpy as np
import sys
import os
from sentence_transformers import SentenceTransformer
from logistic_regression_model import load_model

def main():
    print("=== Prompt Injection Detection Demo ===\n")
    
    print("Loading embedding model...")
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[✓] Embedding model loaded successfully!\n")
    
    except Exception as e:
        print(f"[✗] Error loading embedding model: {e}")
        return
    
    model_path = os.path.join(os.path.join(os.path.dirname(__file__), 'logistic_regression_model.pkl'))
    if not os.path.exists(model_path):
        print(f"[✗] Model file not found: {model_path}")
        return
    
    try:
        lr_model = load_model(model_path)
        print("[✓] Logistic regression model loaded successfully!\n")
        
        if not lr_model.is_trained:
            print("[✗] The loaded model hasn't been trained yet!")
            return
        
        print("=== Model Information ===")
        print(f"Learning rate: {lr_model.learning_rate}")
        print(f"Number of iterations: {lr_model.num_iterations}")
        print(f"Is trained: {lr_model.is_trained}")
        print(f"Weight shape: {lr_model.weights.shape if lr_model.weights is not None else 'None'}")
        print(f"Bias: {lr_model.bias}")
        print(f"Number of training costs recorded: {len(lr_model.costs)}")

        if lr_model.costs:
            print(f"Final training cost: {lr_model.costs[-1]:.6f}")
        
    except Exception as e:
        print(f"[✗] Error loading model: {e}")
        return
    
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
            
            embedding = embedding_model.encode([user_input])
            X_input = embedding.T
            
            prediction = lr_model.predict(X_input)
            probability = lr_model.predict_proba(X_input)
            
            is_malicious = prediction[0, 0] == 1
            confidence = probability[0, 0] if is_malicious else 1 - probability[0, 0]
            
            print(f"\n--- Results ---")
            print(f"Classification: {'[!] MALICIOUS' if is_malicious else '[✓] LEGITIMATE'}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Raw probability (malicious): {probability[0, 0]:.4f}")
            print("-" * 40 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error processing input: {e}\n")

if __name__ == "__main__":
    main()