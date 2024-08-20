import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from losses.loss import SelfConfidMSELoss
from torch.nn import functional as F

# Custom
import config
from models.query_model import VAE_MR, Discriminator_MR, Confidnet, Discriminator_Rfor2
from sampler.sampler import SubsetSequentialSampler


def read_data(dataloader):
    while True:
        for img, label, _ in dataloader:
            yield img, label


# mse + KL
def vae_loss(x, recon, mu, logvar, beta):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD


def train_baVaal(args, models, optimizers, labeled_dataloader, unlabeled_dataloader, cycle):
    'with ranker and Confid net'
    vae = models['vae']
    discriminator = models['discriminator']
    task_model = models['backbone']
    td_predicter = models['module']
    # confidnet = models['confidnet']

    task_model.eval()
    td_predicter.eval()
    vae.train()
    discriminator.train()
    # confidnet.train()
    with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
        vae = vae.cuda()
        discriminator = discriminator.cuda()
        task_model = task_model.cuda()
        td_predicter = td_predicter.cuda()
        # confidnet = confidnet.cuda()

    adversary_param = 1
    beta = 1
    num_adv_steps = 1
    num_vae_steps = 1
    num_confid_steps = 1
    with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
        ce_loss = nn.CrossEntropyLoss().cuda()
        # uncertain_mse_loss = SelfConfidMSELoss(1.0, torch.device('cuda'))
        # mse_loss = nn.MSELoss().cuda()
    labeled_data = read_data(labeled_dataloader)
    unlabeled_data = read_data(unlabeled_dataloader)
    train_iterations = int((args.add_num * cycle + args.subset) * config.EPOCHV / config.BATCH_SIZE)
    # batch 训练
    for iter_count in range(train_iterations):
        labeled_imgs, labels = next(labeled_data)
        unlabeled_imgs = next(unlabeled_data)[0]

        with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
            labeled_imgs = labeled_imgs.cuda()
            unlabeled_imgs = unlabeled_imgs.cuda()
            labels = labels.cuda()

        if iter_count == 0:
            with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
                r_l_0 = torch.from_numpy(np.random.uniform(0, 1, size=(labeled_imgs.shape[0], 10))).type(
                    torch.FloatTensor).cuda()
                r_u_0 = torch.from_numpy(np.random.uniform(0, 1, size=(unlabeled_imgs.shape[0], 10))).type(
                    torch.FloatTensor).cuda()
        else:
            with torch.no_grad():
                _, _, features_l = task_model(labeled_imgs)
                _, _, feature_u = task_model(unlabeled_imgs)
                r_l = td_predicter(features_l)
                r_u = td_predicter(feature_u)

        if iter_count == 0:
            r_l = r_l_0.detach()
            r_u = r_u_0.detach()
            r_l_s = r_l_0.detach()
            r_u_s = r_u_0.detach()
        else:
            r_l_s = torch.softmax(r_l, dim=1).detach()
            r_u_s = torch.softmax(r_u, dim=1).detach()

        # VAE step
        for count in range(num_vae_steps):  # num_vae_steps
            recon, _, mu, logvar = vae(r_l_s, labeled_imgs)
            #
            labeled_vae_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta)
            unlab_recon, _, unlab_mu, unlab_logvar = vae(r_u_s, unlabeled_imgs)
            unlabeled_vae_loss = vae_loss(unlabeled_imgs,
                                          unlab_recon, unlab_mu, unlab_logvar, beta)

            labeled_preds = discriminator(r_l, mu)
            unlabeled_preds = discriminator(r_u, unlab_mu)

            # vae训练隐空间混淆 0为已标注
            lab_real_targets = torch.zeros(labeled_imgs.size(0)).long()
            unlab_fake_targets = torch.zeros(unlabeled_imgs.size(0)).long()

            with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
                lab_real_targets = lab_real_targets.cuda()
                unlab_fake_targets = unlab_fake_targets.cuda()

            # 对抗判别器loss
            dis_loss = ce_loss(labeled_preds, lab_real_targets) + \
                       ce_loss(unlabeled_preds, unlab_fake_targets)
            total_vae_loss = labeled_vae_loss + unlabeled_vae_loss + adversary_param * dis_loss

            optimizers['vae'].zero_grad()
            total_vae_loss.backward()
            optimizers['vae'].step()

            # sample new batch if needed to train the adversarial network
            # if count < (num_vae_steps - 1):
            #     labeled_imgs, _ = next(labeled_data)
            #     unlabeled_imgs = next(unlabeled_data)[0]
            #
            #     with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
            #         labeled_imgs = labeled_imgs.cuda()
            #         unlabeled_imgs = unlabeled_imgs.cuda()
            #         labels = labels.cuda()

        # Discriminator step
        for count in range(num_adv_steps):
            with torch.no_grad():
                _, _, mu, _ = vae(r_l_s, labeled_imgs)
                _, _, unlab_mu, _ = vae(r_u_s, unlabeled_imgs)

            labeled_preds = discriminator(r_l, mu)
            unlabeled_preds = discriminator(r_u, unlab_mu)

            lab_real_targets = torch.zeros(labeled_imgs.size(0)).long()
            unlab_real_targets = torch.ones(unlabeled_imgs.size(0)).long()

            with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
                lab_real_targets = lab_real_targets.cuda()
                unlab_real_targets = unlab_real_targets.cuda()

            dis_loss = ce_loss(labeled_preds, lab_real_targets) + \
                       ce_loss(unlabeled_preds, unlab_real_targets)

            optimizers['discriminator'].zero_grad()
            dis_loss.backward()
            optimizers['discriminator'].step()

            # sample new batch if needed to train the adversarial network
            # if count < (num_adv_steps - 1):
            #     labeled_imgs, _ = next(labeled_data)
            #     unlabeled_imgs = next(unlabeled_data)[0]
            #
            #     with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
            #         labeled_imgs = labeled_imgs.cuda()
            #         unlabeled_imgs = unlabeled_imgs.cuda()
            #         labels = labels.cuda()
        # for count in range(num_confid_steps):
        #     with torch.no_grad():
        #         _, _, mu, _ = vae(r_l_s, labeled_imgs)
        #         _, _, unlab_mu, _ = vae(r_u_s, unlabeled_imgs)
        #
        #         labeled_preds = discriminator(r_l, mu)
        #         unlabeled_preds = discriminator(r_u, unlab_mu)
        #
        #     lab_uncertain = confidnet(r_l, mu)
        #     unlab_uncertain = confidnet(r_u, unlab_mu)
        #     # uncertain_mse_loss need long()
        #     # lab_targets = torch.zeros(labeled_imgs.size(0)).long()
        #     # unlab_targets = torch.ones(unlabeled_imgs.size(0)).long()
        #
        #     lab_targets = torch.zeros(labeled_imgs.size(0)).view(-1,1)
        #     unlab_targets = torch.ones(unlabeled_imgs.size(0)).view(-1,1)
        #     with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
        #         lab_targets = lab_targets.cuda()
        #         unlab_targets = unlab_targets.cuda()
        #
        #     # uncertain_loss = uncertain_mse_loss(labeled_preds, lab_uncertain, lab_targets) \
        #     #                  + uncertain_mse_loss(unlabeled_preds, unlab_uncertain, unlab_targets)
        #     uncertain_loss = mse_loss(lab_uncertain,lab_targets) + mse_loss(unlab_uncertain,unlab_targets)
        #     optimizers['confidnet'].zero_grad()
        #     uncertain_loss.backward()
        #     optimizers['confidnet'].step()

def train_baVaalConfid(args, models, optimizers, labeled_dataloader, unlabeled_dataloader, cycle):
    'with ranker and Confid net'
    vae = models['vae']
    discriminator = models['discriminator']
    task_model = models['backbone']
    td_predicter = models['module']
    confidnet = models['confidnet']

    task_model.eval()
    td_predicter.eval()
    vae.train()
    discriminator.train()
    confidnet.train()
    with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
        vae = vae.cuda()
        discriminator = discriminator.cuda()
        task_model = task_model.cuda()
        td_predicter = td_predicter.cuda()
        confidnet = confidnet.cuda()

    adversary_param = 1
    beta = 1
    num_adv_steps = 1
    num_vae_steps = 1
    num_confid_steps = 1
    with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
        ce_loss = nn.CrossEntropyLoss().cuda()
        uncertain_mse_loss = SelfConfidMSELoss(1.0, torch.device('cuda'))
        # mse_loss = nn.MSELoss().cuda()
    labeled_data = read_data(labeled_dataloader)
    unlabeled_data = read_data(unlabeled_dataloader)
    train_iterations = int((args.add_num * cycle + args.subset) * config.EPOCHV / config.BATCH_SIZE)
    # batch 训练
    for iter_count in range(train_iterations):
        labeled_imgs, labels = next(labeled_data)
        unlabeled_imgs = next(unlabeled_data)[0]

        with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
            labeled_imgs = labeled_imgs.cuda()
            unlabeled_imgs = unlabeled_imgs.cuda()
            labels = labels.cuda()

        if iter_count == 0:
            with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
                r_l_0 = torch.from_numpy(np.random.uniform(0, 1, size=(labeled_imgs.shape[0], 10))).type(
                    torch.FloatTensor).cuda()
                r_u_0 = torch.from_numpy(np.random.uniform(0, 1, size=(unlabeled_imgs.shape[0], 10))).type(
                    torch.FloatTensor).cuda()
        else:
            with torch.no_grad():
                _, _, features_l = task_model(labeled_imgs)
                _, _, feature_u = task_model(unlabeled_imgs)
                r_l = td_predicter(features_l)
                r_u = td_predicter(feature_u)

        if iter_count == 0:
            r_l = r_l_0.detach()
            r_u = r_u_0.detach()
            r_l_s = r_l_0.detach()
            r_u_s = r_u_0.detach()
        else:
            r_l_s = torch.softmax(r_l, dim=1).detach()
            r_u_s = torch.softmax(r_u, dim=1).detach()

        # VAE step
        for count in range(num_vae_steps):  # num_vae_steps
            recon, _, mu, logvar = vae(r_l_s, labeled_imgs)
            #
            labeled_vae_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta)
            unlab_recon, _, unlab_mu, unlab_logvar = vae(r_u_s, unlabeled_imgs)
            unlabeled_vae_loss = vae_loss(unlabeled_imgs,
                                          unlab_recon, unlab_mu, unlab_logvar, beta)

            labeled_preds = discriminator(r_l, mu)
            unlabeled_preds = discriminator(r_u, unlab_mu)

            # vae训练隐空间混淆 0为已标注
            lab_real_targets = torch.zeros(labeled_imgs.size(0)).long()
            unlab_fake_targets = torch.zeros(unlabeled_imgs.size(0)).long()

            with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
                lab_real_targets = lab_real_targets.cuda()
                unlab_fake_targets = unlab_fake_targets.cuda()

            # 对抗判别器loss
            dis_loss = ce_loss(labeled_preds, lab_real_targets) + \
                       ce_loss(unlabeled_preds, unlab_fake_targets)
            total_vae_loss = labeled_vae_loss + unlabeled_vae_loss + adversary_param * dis_loss

            optimizers['vae'].zero_grad()
            total_vae_loss.backward()
            optimizers['vae'].step()

            # sample new batch if needed to train the adversarial network
            # if count < (num_vae_steps - 1):
            #     labeled_imgs, _ = next(labeled_data)
            #     unlabeled_imgs = next(unlabeled_data)[0]
            #
            #     with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
            #         labeled_imgs = labeled_imgs.cuda()
            #         unlabeled_imgs = unlabeled_imgs.cuda()
            #         labels = labels.cuda()

        # Discriminator step
        for count in range(num_adv_steps):
            with torch.no_grad():
                _, _, mu, _ = vae(r_l_s, labeled_imgs)
                _, _, unlab_mu, _ = vae(r_u_s, unlabeled_imgs)

            labeled_preds = discriminator(r_l, mu)
            unlabeled_preds = discriminator(r_u, unlab_mu)

            lab_real_targets = torch.zeros(labeled_imgs.size(0)).long()
            unlab_real_targets = torch.ones(unlabeled_imgs.size(0)).long()

            with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
                lab_real_targets = lab_real_targets.cuda()
                unlab_real_targets = unlab_real_targets.cuda()

            dis_loss = ce_loss(labeled_preds, lab_real_targets) + \
                       ce_loss(unlabeled_preds, unlab_real_targets)

            optimizers['discriminator'].zero_grad()
            dis_loss.backward()
            optimizers['discriminator'].step()

            # sample new batch if needed to train the adversarial network
            # if count < (num_adv_steps - 1):
            #     labeled_imgs, _ = next(labeled_data)
            #     unlabeled_imgs = next(unlabeled_data)[0]
            #
            #     with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
            #         labeled_imgs = labeled_imgs.cuda()
            #         unlabeled_imgs = unlabeled_imgs.cuda()
            #         labels = labels.cuda()
        for count in range(num_confid_steps):
            with torch.no_grad():
                _, _, mu, _ = vae(r_l_s, labeled_imgs)
                _, _, unlab_mu, _ = vae(r_u_s, unlabeled_imgs)

                labeled_preds = discriminator(r_l, mu)
                unlabeled_preds = discriminator(r_u, unlab_mu)

            lab_uncertain = confidnet(r_l, mu)
            unlab_uncertain = confidnet(r_u, unlab_mu)
            # uncertain_mse_loss need long()
            lab_targets = torch.zeros(labeled_imgs.size(0)).long()
            unlab_targets = torch.ones(unlabeled_imgs.size(0)).long()

            # lab_targets = torch.zeros(labeled_imgs.size(0)).view(-1,1)
            # unlab_targets = torch.ones(unlabeled_imgs.size(0)).view(-1,1)
            with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
                lab_targets = lab_targets.cuda()
                unlab_targets = unlab_targets.cuda()

            uncertain_loss = uncertain_mse_loss(labeled_preds, lab_uncertain, lab_targets) \
                              + uncertain_mse_loss(unlabeled_preds, unlab_uncertain, unlab_targets)
            # uncertain_loss = mse_loss(lab_uncertain,lab_targets) + mse_loss(unlab_uncertain,unlab_targets)
            optimizers['confidnet'].zero_grad()
            uncertain_loss.backward()
            optimizers['confidnet'].step()


# Select the indices of the unlabeled data according to the methods
def query_samples(model, data_unlabeled, subset, labeled_set, cycle, args, ratio):
    # Create unlabeled dataloader for the unlabeled subset
    unlabeled_loader = DataLoader(data_unlabeled, batch_size=config.BATCH_SIZE,
                                  sampler=SubsetSequentialSampler(subset),
                                  pin_memory=True)
    labeled_loader = DataLoader(data_unlabeled, batch_size=config.BATCH_SIZE,
                                sampler=SubsetSequentialSampler(labeled_set),
                                pin_memory=True)
    if args.dataset == 'fashionmnist':
        vae = VAE_MR(28, 1, 3)
        discriminator = Discriminator_Rfor2(28, 10)
        # confidnet = Confidnet(28, 10)
    else:
        vae = VAE_MR()
        discriminator = Discriminator_Rfor2(32, 10)
        # confidnet = Confidnet(32, 10)

    models = {'backbone': model['backbone'],
              'module': model['module'],
              'vae': vae,
              'discriminator': discriminator}

    optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
    # optim_confidnet = optim.Adam(confidnet.parameters(), lr=5e-4)
    optimizers = {'vae': optim_vae,
                  'discriminator': optim_discriminator,
                  }

    train_baVaal(args, models, optimizers, labeled_loader, unlabeled_loader, cycle + 1)
    print("Finished train semi-modules")
    task_model = models['backbone']
    td_predicter = models['module']
    all_preds, all_indices = [], []

    for images, fake_targets, indices in unlabeled_loader:
        with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
            images = images.cuda()
        with torch.no_grad():
            out, _, features = task_model(images)
            td = td_predicter(features)
            images_recon, _, mu, _ = vae(torch.softmax(td, dim=1), images)
            # loss = recon_loss(images_recon, images)
            # loss = torch.sum(loss, dim=(1, 2, 3)) / 3072
            # for i in range(len(fake_targets)):
            #     dict1[fake_targets[i].item()] += 1
            #     dict2[fake_targets[i].item()] += loss[i].item()
            preds = discriminator(td, mu)
            preds = F.softmax(preds, dim=1)
            preds = preds[:, 1]
            preds = preds.cpu()
            # uncertain相关
            # uncertain = confidnet(td, mu)
            # uncertain = torch.sigmoid(uncertain)
            # uncertain = uncertain[:, 0]
            # uncertain = uncertain.cpu()
            # 使用ratio修改preds
            ratio_tensor = torch.ones_like(preds)
            for i in range(ratio_tensor.shape[0]):
                for j in range(10):
                    ratio_tensor[i] = torch.where(fake_targets[i].data == j, ratio_tensor[i] * (1 + ratio[j]),
                                                  ratio_tensor[i])

            preds = preds * ratio_tensor


        preds = preds.data
        all_preds.extend(preds)
        all_indices.extend(indices)
    # print(dict1)
    # print(dict2)
    all_preds = torch.stack(all_preds)
    all_preds = all_preds.view(-1)
    # need to multiply by -1 to be able to use torch.topk
    # select the points which the discriminator things are the most likely to be unlabeled
    _, arg = torch.sort(all_preds)


    return arg


def query_samples_Confid(model, data_unlabeled, subset, labeled_set, cycle, args, ratio):
    # Create unlabeled dataloader for the unlabeled subset
    unlabeled_loader = DataLoader(data_unlabeled, batch_size=config.BATCH_SIZE,
                                  sampler=SubsetSequentialSampler(subset),
                                  pin_memory=True)
    labeled_loader = DataLoader(data_unlabeled, batch_size=config.BATCH_SIZE,
                                sampler=SubsetSequentialSampler(labeled_set),
                                pin_memory=True)
    if args.dataset == 'fashionmnist':
        vae = VAE_MR(28, 1, 3)
        discriminator = Discriminator_Rfor2(28, 10)
        confidnet = Confidnet(28, 10)
    else:
        vae = VAE_MR()
        discriminator = Discriminator_Rfor2(32, 10)
        confidnet = Confidnet(32, 10)

    models = {'backbone': model['backbone'],
              'module': model['module'],
              'vae': vae,
              'discriminator': discriminator,
              'confidnet': confidnet}

    optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
    optim_confidnet = optim.Adam(confidnet.parameters(), lr=5e-4)
    optimizers = {'vae': optim_vae,
                  'discriminator': optim_discriminator,
                  'confidnet': optim_confidnet}

    train_baVaalConfid(args, models, optimizers, labeled_loader, unlabeled_loader, cycle + 1)
    print("Finished train semi-modules")
    task_model = models['backbone']
    td_predicter = models['module']
    all_preds, all_indices = [], []
    file_name1 = r'logs/ours_prob.txt'
    prob_file1 = open(file_name1, 'w+')
    file_name2 = r'logs/ours_withu_prob.txt'
    prob_file2 = open(file_name2,'w+')
    for images, fake_targets, indices in unlabeled_loader:
        with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
            images = images.cuda()
        with torch.no_grad():
            out, _, features = task_model(images)
            td = td_predicter(features)
            images_recon, _, mu, _ = vae(torch.softmax(td, dim=1), images)
            # loss = recon_loss(images_recon, images)
            # loss = torch.sum(loss, dim=(1, 2, 3)) / 3072
            # for i in range(len(fake_targets)):
            #     dict1[fake_targets[i].item()] += 1
            #     dict2[fake_targets[i].item()] += loss[i].item()
            preds = discriminator(td, mu)
            preds = F.softmax(preds, dim=1)
            preds = preds[:, 1]
            preds = preds.cpu()
            np.array(preds).tofile(prob_file1, sep='\n')
            prob_file1.write('\n')
            # uncertain相关
            uncertain = confidnet(td, mu)
            uncertain = torch.sigmoid(uncertain)
            uncertain = uncertain[:, 0]
            uncertain = uncertain.cpu()
            # 使用ratio修改preds
            ratio_tensor = torch.ones_like(preds)
            for i in range(ratio_tensor.shape[0]):
                for j in range(10):
                    ratio_tensor[i] = torch.where(fake_targets[i].data == j, ratio_tensor[i] * (1 + ratio[j]),
                                                  ratio_tensor[i])
            preds = preds * uncertain
            # preds = uncertain
            np.array(preds).tofile(prob_file2,sep='\n')
            prob_file2.write('\n')
            preds = preds * ratio_tensor


        preds = preds.data
        all_preds.extend(preds)
    all_indices.extend(indices)
    # print(dict1)
    # print(dict2)
    all_preds = torch.stack(all_preds)
    all_preds = all_preds.view(-1)
    # need to multiply by -1 to be able to use torch.topk
    # select the points which the discriminator things are the most likely to be unlabeled
    _, arg = torch.sort(all_preds)

    prob_file1.close()
    prob_file2.close()

    return arg


def update_fake_unlabeled_dataset(models, fake_unlabeled_loader):
    """
        更新一未标记数据池的伪标签dataset,使其送入下一次vae训练中获得伪标签；
    """
    models['backbone'].eval()
    with torch.no_grad():

        for data in fake_unlabeled_loader:
            new_fake_labels = []
            inputs, targets, indices = data
            with torch.cuda.device(config.CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
            scores, _, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)

            new_fake_labels.extend(preds.cpu().tolist())

            for i in range(len(indices)):
                fake_unlabeled_loader.dataset.dataset.targets[indices[i]] = new_fake_labels[i]
