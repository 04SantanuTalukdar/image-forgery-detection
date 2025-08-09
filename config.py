class Config:
    data_dir = "./dataset"         
    batch_size = 16             
    num_epochs = 30
    learning_rate = 0.0005
    device = "cuda"
    early_stopping_patience = 5
    model_save_path = "./best_forgery_model.pth"
    image_size = 256           
