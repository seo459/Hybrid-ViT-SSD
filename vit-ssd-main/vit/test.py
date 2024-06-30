def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total}%')


if __name__ == "__main__":
    # Assume the following parameters
    kwargs = {
        'p': 16,
        'model_dim': 768,
        'hidden_dim': 3072,
        'n_class': 1000,
        'n_heads': 12,
        'n_layers': 12,
        'n_patches': 196,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT(**kwargs).to(device)

    # Load the trained model
    model.load_state_dict(torch.load("vit_model.pth"))

    # Assume test_loader is prepared
    test_loader = ...

    # Test the model
    test(model, test_loader, device)
