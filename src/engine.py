import torch
from torch.cuda.amp import GradScaler, autocast
from src.utils import visualize_results
from src.loss import SteganoLoss

def train_step(model: torch.nn.Module,
                loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer, 
                criterion: SteganoLoss, 
                device: torch.device, 
                scaler: torch.cuda.amp.GradScaler, 
                accum_steps):
    model.train()
    running_loss = 0.0
    
    for i, (cover, secret, edge, _) in enumerate(loader):
        cover, secret, edge = cover.to(device), secret.to(device), edge.to(device)

        with autocast():
            stego, feat, fused = model.hide_net(cover, secret, edge)
            recovered = model.reveal_net(stego, edge, feat, fused)
            loss, _ = criterion(stego, cover, recovered, secret, edge)
            loss = loss / accum_steps

        scaler.scale(loss).backward()
        
        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        running_loss += loss.item() * accum_steps
        
    return running_loss / len(loader)

def validate_step(model: torch.nn.Module,
                loader: torch.utils.data.DataLoader,
                criterion: SteganoLoss, 
                device: torch.device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for cover, secret, edge, _ in loader:
            cover, secret, edge = cover.to(device), secret.to(device), edge.to(device)
            stego, feat, fused = model.hide_net(cover, secret, edge)
            recovered = model.reveal_net(stego, edge, feat, fused)
            loss, _ = criterion(stego, cover, recovered, secret, edge)
            val_loss += loss.item()
    return val_loss / len(loader)

def train(model: torch.nn.Module,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            criterion: SteganoLoss,
            config: dict,
            device: torch.device):
    """Main orchestration function for the training process."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_loss = float('inf')
    
    for epoch in range(config['epochs']):
        train_loss = train_step(model, train_loader, optimizer, criterion, device, scaler, config['accum_steps'])
        val_loss = validate_step(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), config['model_path'])
            print("→ Checkpoint Saved")
            
        if epoch % 5 == 0:
            visualize_results(model, val_loader, device)