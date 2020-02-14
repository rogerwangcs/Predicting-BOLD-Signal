from tqdm import tqdm_notebook


def fitAE(model, criterion, optimizer, dataloader, epochs=10):
    model.train()
    loss_memory = []
    pbar = tqdm_notebook(range(epochs), leave=True)

    numBatches = len(dataloader)
    # set running loss print frequency
    lossCutoff = min(numBatches, (numBatches*epochs)/100)

    for epoch in pbar:  # loop over the dataset multiple times
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            (inputBatch, labelBatch) = batch
            inputBatch = inputBatch.cuda()
            labelBatch = labelBatch.cuda()

            # forward
            optimizer.zero_grad()
            outputBatch = model(inputBatch.float())[0]
            loss = criterion(outputBatch, inputBatch)

            # backward
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data.item()
            if i % lossCutoff == 0:
                pbar.set_description("Loss: %.4f" % (running_loss))
                pbar.refresh()
                running_loss = 0.0


def fitFeatureModel(model, criterion, optimizer, dataloader, epochs=10):
    model.train()
    loss_memory = []
    pbar = tqdm_notebook(range(epochs), leave=True)

    numBatches = len(dataloader)
    # set running loss print frequency
    lossCutoff = min(numBatches, (numBatches*epochs)/100)

    for epoch in pbar:  # loop over the dataset multiple times
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            (inputBatch, labelBatch) = batch
            inputBatch = inputBatch.cuda()
            labelBatch = labelBatch.cuda()

            # forward
            optimizer.zero_grad()
            outputBatch = model(inputBatch.float())
            loss = criterion(outputBatch, labelBatch.float())

            # backward
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data.item()
            if i % lossCutoff == 0:
                pbar.set_description("Loss: %.4f" % (running_loss))
                pbar.refresh()
                running_loss = 0.0
