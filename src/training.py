import torch


# Обучение одной эпохи
def train_one_epoch(epoch_index, model, train_loader, optimizer, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    last_loss = 0.
    running_loss = 0.

    # Обновление весов модели и расчёт ошибки 
    for batch_index, data in enumerate(train_loader):
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_index % 20 == 19:
            last_loss = running_loss / 20. # средняя ошибка за 20 батчей
            print(f'Эпоха: {epoch_index}, батч: {batch_index}, ошибка {last_loss}')
            running_loss = 0.

    return last_loss

# Полынй цикл обучения
def training(EPOCHS, model, train_loader, val_loader, optimizer, criterion):
    
    best_vloss = 1e5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(EPOCHS):
        print(f'Эпоха {epoch}')

        # Обучение эпохи и подсчет ошибки на валидации
        model.to(device)
        model.train(True)

        avg_loss = train_one_epoch(epoch, model, train_loader, optimizer, criterion)

        model.eval()

        running_vloss = 0.0
        # Валидация
        with torch.no_grad():
            for vindex, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (vindex + 1)
        
        # Сохранение лучшей модели
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'models/meds_classifier_{epoch}.pt'
            torch.save(model.state_dict(), model_path)

        print(f'В конце эпохи ошибка train {avg_loss}, ошибка val {avg_vloss}')
