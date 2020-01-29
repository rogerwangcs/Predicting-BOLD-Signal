from tqdm import tqdm_notebook


def fitAE(model, criterion, optimizer, dataloader, total_epochs=10):
    model.train()
    loss_memory = []
    pbar = tqdm_notebook(range(total_epochs), leave=True)
    for epoch in pbar:  # loop over the dataset multiple times
        running_loss = 0.0
        for i, sample in enumerate(dataloader):
            (inputBatch, labelBatch) = sample
            inputBatch = inputBatch.cuda()

            # forward
            optimizer.zero_grad()
            outputBatch = model(inputBatch.float())[0]
            # check difference between original img vs output img
            loss = criterion(outputBatch, inputBatch)

            # backward
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data.item()
            if i % 50 == 49:
                pbar.set_description("Loss: %.4f" % (running_loss))
                pbar.refresh()
                running_loss = 0.0
