
def zero_padding(arr, patchsize):
    # print("zero_padding patchsize: {}".format(patchsize))
    (h, w, c) = arr.shape
    pad_h = (1 + np.floor(h/patchsize)) * patchsize - h
    pad_w = (1 + np.floor(w/patchsize)) * patchsize - w

    arr_pad = np.pad(arr, ((0, int(pad_h)), (0, int(pad_w)), (0, 0)), mode='symmetric')
    return arr_pad

def model_inference(model, file, savePath):
    # model.cpu()
    patchsize = 512
    img0 = tiff.imread(file)

    # img = (np.nan_to_num(img0, 0) + 1) * 255 / 2
    img = interval_95(np.nan_to_num(img0, 0)) * 255
    # # img = np.nan_to_num(img0, 0)
    # # img = cv2.cvtColor(img1.transpose(1,2,0), cv2.COLOR_BGR2RGB)

    input_patchsize = 2 * patchsize
    padSize = int(patchsize/2)

    H, W, C = img.shape
    img_pad0 = zero_padding(img, patchsize) # pad img into a shape: (m*PATCHSIZE, n*PATCHSIZE)
    img_pad = np.pad(img_pad0, ((padSize, padSize), (padSize, padSize), (0, 0)), mode='symmetric')

    img_preprocessed = preprocessing_fn(img_pad)
    in_tensor = torch.from_numpy(img_preprocessed.transpose(2, 0, 1)).unsqueeze(0)

    (Height, Width, Channels) = img_pad.shape
    pred_mask_pad = np.zeros((Height, Width))
    for i in tqdm(range(0, Height - input_patchsize + 1, patchsize)):
        for j in range(0, Width - input_patchsize + 1, patchsize):
            # print(i, i+input_patchsize, j, j+input_patchsize)

            inputPatch = in_tensor[..., i:i+input_patchsize, j:j+input_patchsize]

            if ARCH == 'FCN':
                predPatch = model(inputPatch.type(torch.cuda.FloatTensor))['out']
            else:
                predPatch = model(inputPatch.type(torch.cuda.FloatTensor))

            predPatch = predPatch.squeeze().cpu().detach().numpy()#.round()
            pred_mask_pad[i+padSize:i+padSize+patchsize, j+padSize:j+padSize+patchsize] = predPatch[padSize:padSize+patchsize, padSize:padSize+patchsize]  # need to modify

    pred_mask = pred_mask_pad[padSize:padSize+H, padSize:padSize+W] # clip back to original shape

    if True:
        dataName = os.path.split(file)[-1][:-4]
        # imsave(SAVEPATH / f'{dataName}_input.png', (img).astype(np.uint8))
        # imsave(savePath / f"conf_{dataName}.png", (pred_mask*255).astype(np.uint8))
        imsave(savePath / f"pred_{dataName}.png", (pred_mask.round()*255).astype(np.uint8))


    if False:
        plt.figure(figsize=(10,10))
        plt.subplot(121)
        # plt.imshow(interval_95(img_preprocessed[:H, :W]), cmap='gray')
        plt.imshow(interval_95(img), cmap='gray')
        plt.subplot(122)
        plt.imshow(pred_mask, cmap='gray')
        
    return pred_mask