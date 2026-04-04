import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import BasicPreprocessing
from model import ModelDevelopment, ModelTrainer
from inference import BasicInference


def main():
    # Config
    DATA_DIR   = 'data'
    IMG_SIZE   = 128
    BATCH_SIZE = 32
    EPOCHS     = 30
    LR         = 0.001
    MODEL_PATH = 'models/best_model.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs('models',  exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Task 1: Preprocessing
    print("\n[Task 1] Loading and preprocessing data.")
    preprocessor = BasicPreprocessing(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    train_loader, val_loader, test_loader = preprocessor.get_dataloaders()

    # Visualize some samples
    image_paths, labels = preprocessor.import_dataset()
    preprocessor.visualize_samples(image_paths, labels)

    # Task 2: Build & Train Model
    print("\n[Task 2] Building model.")
    model = ModelDevelopment(num_classes=2)
    summary = model.get_architecture_summary()
    print(f"Total parameters: {summary['total_params']:,}")

    trainer = ModelTrainer(model, device, learning_rate=LR)
    history = trainer.train(train_loader, val_loader,
                            epochs=EPOCHS, save_path=MODEL_PATH)
    trainer.plot_history()

    # Task 3: Evaluate
    print("\n[Task 3] Evaluating on test set...")
    # Load best saved model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    inferencer = BasicInference(model, device, IMG_SIZE,
                                classes=['with_mask', 'without_mask'])
    inferencer.evaluate_on_test_set(test_loader)

main()