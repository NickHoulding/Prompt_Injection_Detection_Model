import tensorflow as tf
import numpy as np
import ollama
import os

def main():
    print("=== Prompt Injection Detection Demo ===\n")

    model_path = os.path.join(os.path.dirname(__file__), 'models', 'nn_model.keras')
    if not os.path.exists(model_path):
        print(f"[✗] Model file not found: {model_path}")
        return
    
    try:
        nn_model = tf.keras.models.load_model(model_path)
        print("[✓] Neural network model loaded successfully!\n")

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
            
            response = ollama.embed(
                model='nomic-embed-text',
                input=user_input,
            )
            X_input = np.array(response['embeddings']).reshape(1, -1)

            prediction = nn_model.predict(X_input, verbose=0)
            
            # The model outputs a probability between 0 and 1
            malicious_probability = prediction[0, 0]
            is_malicious = malicious_probability > 0.5
            confidence = malicious_probability if is_malicious else 1 - malicious_probability
            
            print(f"\n--- Results ---")
            print(f"Classification: {'[!] MALICIOUS' if is_malicious else '[✓] LEGITIMATE'}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Raw probability (malicious): {malicious_probability:.4f}")
            print("-" * 40 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error processing input: {e}\n")

if __name__ == "__main__":
    main()