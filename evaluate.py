"""
    Learn2Perturb evaluation function
"""

import torch
from tqdm.auto import tqdm

from attacks.pgd import pgd
from attacks.fgsm import fgsm
from attacks.eot import EOT_FGSM, EOT_PGD

from torchattacks import *

def evaluate_robust(model, subset_loader):
    model.eval()

    attacks = {
        'FGSM': FGSM(model, eps=8 / 255),
        # 'PGD': PGD(model, eps=8 / 255, alpha=2 / 225, steps=100, random_start=True),
        # 'PGD-EOT': EOTPGD(model, eps=8 / 255, alpha=2 / 255, steps=100, eot_iter=2)
    }

    for attack_name, attack in attacks.items():
        correct, total = 0, 0
        for images, labels in tqdm(subset_loader, leave=False):
            images, labels = images.cuda(), labels.cuda()

            adv_images = attack(images, labels)
            outputs = model(adv_images)
            _, pred = torch.max(outputs.data, 1)
            total += len(images)
            correct += (pred == labels).sum()

        print(f'        {attack_name} robust accuracy: {(100. * correct / total)}%')

def evaluate(model, testloader, attack=None, epsilon=8 / 255):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(testloader, leave=False):
            inputs, labels = data[0], data[1]
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            if attack == 'pgd':
                perturbed_image = pgd(model, inputs, labels, epsilon=epsilon)
                outputs = model(perturbed_image)
                input_info = "Accuracy under pgd attack with epsilon = " + str(epsilon)
            elif attack == 'fgsm':
                perturbed_image = fgsm(model, inputs, labels, epsilon=epsilon)
                outputs = model(perturbed_image)
                input_info = "Accuracy under fgsm attack with epsilon = " + str(epsilon)
            elif attack == 'EOT-fgsm':
                perturbed_image = EOT_FGSM(model, inputs, labels, epsilon=epsilon)
                outputs = model(perturbed_image)
                input_info = "Accuracy under EOT-fgsm attack with epsilon = " + str(epsilon)
            elif attack == 'EOT-pgd':
                perturbed_image = EOT_PGD(model, inputs, labels, epsilon=epsilon)
                outputs = model(perturbed_image)
                input_info = "Accuracy under EOT-pgd attack with epsilon = " + str(epsilon)
            elif attack is None:
                outputs = model(inputs)
                input_info = "Clean data accuracy"
            else:
                raise NotImplementedError("Attack not supported!")

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = correct / total
        print('        {} = {}'.format(input_info, 100 * acc))
