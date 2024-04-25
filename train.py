import random
import time
from statistics import mean
import torch
from main import *
from utilities import convert_lab_to_rgb, plot_images, calc_gradient_penalty


def pretrain_generator(epochs=5):
    """ Pre-training ONLY the generator using L1 loss """

    # Random batch to print
    rand_batch = random.randint(0, len(trainloader_color))

    start_time = time.time()
    generator.train()
    for epoch in range(epochs):
        curr_time = time.time()
        running_loss = 0.0

        for batch_idx, orig_color_images in enumerate(trainloader_color):
            optimizer_G.zero_grad()
            L = orig_color_images['L'].to(device)
            orig_ab = orig_color_images['ab'].to(device)

            # Generating the colorized images
            fake_ab = generator(L)

            # Computing the loss between the fake 'ab' and real 'ab'
            loss = loss_function(fake_ab, orig_ab)

            # Accumulating the loss
            running_loss += loss.item()

            # Back propogation
            loss.backward()
            optimizer_G.step()

            # Prints one random batch
            if batch_idx==rand_batch:
                fake_images = convert_lab_to_rgb(L, fake_ab.detach())
                real_images = convert_lab_to_rgb(L, orig_ab)

                # Saving or plotting the current batch of images. In order to plot, set image_dir_path=None
                plot_images(L[:8], fake_images[:8], real_images[:8], epoch, image_dir_path=pretrain_dir, mode='pretrain')

        # For each epoch, print the loss and save the model
        print(f'Epoch {epoch+1}/{epochs}, L1_Loss: {(running_loss / len(trainloader_color)):.4f}')
        print(f"Epoch training time: {((time.time() - curr_time) / 60):.2f} minutes| Total time: {((time.time() - start_time) / 60):.2f} minutes\n")
        torch.save({'model_weights': generator.state_dict(),'optimizer_weights': optimizer_G.state_dict()}, path + 'pretrained_generator.pth')

    print(f'Pre-training completed - Total time is: {((time.time() - start_time) / 60):.2f} minutes')


def train_wgan_gp(epochs, generator_iterations, lambda_gp):
    best_psnr = 0

    start_time = time.time()
    for epoch in range(epochs):
        running_gen_L1_loss, running_gen_WGAN_loss, running_critic_real_loss = [], [], []

        curr_time = time.time()
        generator.train()
        critic.train()

        for batch_idx, orig_color_images in enumerate(trainloader_color):
            # Seperating the channels
            L = orig_color_images['L'].to(device)
            orig_ab = orig_color_images['ab'].to(device)
            real_images = torch.cat([L, orig_ab], dim=1)

            # Train Generator - giving it advantage of generator_iterations
            for p in critic.parameters(): p.requires_grad = False

            for _ in range(generator_iterations):
                optimizer_G.zero_grad()

                # Colorizing the greyscaled images
                fake_ab = generator(L).to(device)
                fake_colorized_images = torch.cat([L, fake_ab], dim=1)
                critic_fake = critic(fake_colorized_images)

                # Comparing the original and fake colorized images
                gen_L1_loss = loss_function(fake_ab, orig_ab)

                # WGAN loss for generator
                gen_WGAN_loss = -torch.mean(critic_fake)

                # Backpropagating the losses
                gen_loss = gen_L1_loss + gen_WGAN_loss
                gen_loss.backward()
                optimizer_G.step()


            # Train Critic
            for p in critic.parameters(): p.requires_grad = True
            optimizer_C.zero_grad()

            # Creating fake images
            fake_ab = generator(L).to(device)
            fake_colorized_images = torch.cat([L, fake_ab], dim=1)

            # Criticizing the images
            critic_fake = critic(fake_colorized_images.detach())
            critic_real = critic(real_images)

            # WGAN-GP loss for critic
            # Computing the gradient penalty
            gp = calc_gradient_penalty(real_images, fake_colorized_images.detach())

            # Computing the critic's loss
            critic_fake_loss = torch.mean(critic_fake)
            critic_real_loss = torch.mean(critic_real)

            # Computing the wasserstein distance
            critic_WGAN_loss = critic_fake_loss - critic_real_loss

            # Backpropagating the losses
            critic_loss = critic_WGAN_loss + lambda_gp*gp
            critic_loss.backward()
            optimizer_C.step()

            # Logging losses 10 times in each epoch
            if batch_idx%(len(trainloader_color)//10) == 0:
                # Generator losses
                gen_L1_losses.append(gen_L1_loss.item())
                gen_WGAN_losses.append(gen_WGAN_loss.item())
                gen_losses.append(gen_loss.item())
                running_gen_L1_loss.append(gen_L1_loss.item())
                running_gen_WGAN_loss.append(gen_WGAN_loss.item())

                # Critic Losses
                critic_real_losses.append(critic_real_loss.item())
                critic_fake_losses.append(critic_fake_loss.item())
                critic_WGAN_losses.append(critic_WGAN_loss.item())
                critic_losses.append(critic_loss.item())
                running_critic_real_loss.append(critic_real_loss.item())

            if batch_idx==len(trainloader_color)-2: # Print status, save images, and validate
                print(f"Summary: Epoch {epoch+1}/{epochs}|  Epoch training time: {((time.time() - curr_time) / 60):.2f} minutes| Total time: {((time.time() - start_time) / 60):.2f} minutes")
                print(f'Training\nAvg Critic Fake Loss: {mean(running_gen_WGAN_loss):.3f}, Avg Critic Real Loss: {mean(running_critic_real_loss):.3f} Avg Generator L1 Loss: {mean(running_gen_L1_loss):.3f}')

                # Converting from Lab to RGB
                fake_images_rgb = convert_lab_to_rgb(L, fake_ab.detach())
                real_images_rgb = convert_lab_to_rgb(L, orig_ab)

                # Saving or plotting the current batch of images. In order to plot, set image_dir_path=None
                plot_images(L[:8], fake_images_rgb[:8], real_images_rgb[:8], epoch, image_dir_path=train_dir, mode='training')

                # Validation
                psnr_val = evaluate_model(epoch, batch_idx)
                generator.train()
                critic.train()


        # Saving the generator's and critic's weights
        if best_psnr < psnr_val:
            best_psnr = psnr_val
            torch.save({'model_weights': generator.state_dict(), 'optimizer_weights': optimizer_G.state_dict()}, path + 'trained_generator.pth')
            torch.save({'model_weights': critic.state_dict(),'optimizer_weights': optimizer_C.state_dict()}, path + 'trained_critic.pth')
        print("_________________________________\n")

    print(f'Training completed - Total time: {((time.time() - start_time) / 60):.2f} miutes')
    test_model(testloader_color)



def evaluate_model(epoch, batch):
    generator.eval()
    critic.eval()

    running_gen_L1_loss, running_gen_WGAN_loss, running_critic_real_loss = [], [], []
    total_psnr = 0.0
    total_samples = 0
    with torch.no_grad():
        for i, orig_color_images in enumerate(validationloader_color):
            # Seperating the channels
            L = orig_color_images['L'].to(device)
            orig_ab = orig_color_images['ab'].to(device)

            # Coloring the images
            fake_ab = generator(L).to(device)
            fake_colorized_images = torch.cat([L, fake_ab], dim=1)
            real_images = torch.cat([L, orig_ab], dim=1)

            # Criticizing the images
            critic_fake = critic(fake_colorized_images)
            critic_real = critic(real_images)

            # Computing the losses
            gen_WGAN_loss = -torch.mean(critic_fake).item()       # WGAN loss for Generator
            critic_real_loss = -torch.mean(critic_real).item()    # WGAN loss for critic real
            gen_L1_loss = loss_function(fake_ab, orig_ab).item()   # L1_Loss of the generator
            gen_loss = gen_L1_loss + gen_WGAN_loss # Generator's loss

            # Computing critic's loss and gradient penalty
            critic_WGAN_loss = critic_real_loss - gen_WGAN_loss

            # Saving the losses for later plotting
            if i%3 == 0:
                # Generator losses
                val_gen_L1_losses.append(gen_L1_loss)
                running_gen_L1_loss.append(gen_L1_loss)
                val_gen_WGAN_losses.append(gen_WGAN_loss)
                running_gen_WGAN_loss.append(-gen_WGAN_loss)
                val_gen_losses.append(gen_loss)

                # Critic losses
                val_critic_real_losses.append(-critic_real_loss)
                running_critic_real_loss.append(-critic_real_loss)
                val_critic_fake_losses.append(-gen_WGAN_loss)
                val_critic_WGAN_losses.append(critic_WGAN_loss)

                # Calculate PSNR for each image in the batch
                batch_psnr = psnr(real_images, fake_colorized_images)
                total_psnr += batch_psnr * L.size(0)
                total_samples += L.size(0)
                psnr_values.append(batch_psnr)  # Append the batch PSNR to the list

            # Saving or plotting the current batch of images. In order to plot, set image_dir_path=None
            batch_num = 8 if epoch%2==0 else 6
            if i==len(validationloader_color)-batch_num:
                fake_images_rgb = convert_lab_to_rgb(L, fake_ab.detach())
                real_images_rgb = convert_lab_to_rgb(L, orig_ab)
                plot_images(L[:8], fake_images_rgb[:8], real_images_rgb[:8], epoch, image_dir_path=val_dir, mode='validation')


    avg_psnr = total_psnr / total_samples
    print(f"\nVALIDATION\nAvg Critic Fake Loss: {mean(running_gen_WGAN_loss):.3f}, Avg Critic Real Loss: {mean(running_critic_real_loss):.3f}, Avg Generator L1 Loss: {mean(running_gen_L1_loss):.3f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")

    return avg_psnr



def test_model(testloader_color):

    if test: # During test, load pre-trained weights
        path=''
        generator.load_state_dict(torch.load(path + 'trained_generator.pth', map_location=device))
    generator.eval()

    total_psnr = 0.0
    total_samples = 0
    rand_batches = random.sample(range(len(testloader_color)), 1)#4)

    with torch.no_grad():
        for batch_idx, orig_color_images in enumerate(testloader_color):
            # Seperating the channels
            L = orig_color_images['L'].to(device)
            orig_ab = orig_color_images['ab'].to(device)

            # Coloring the images
            fake_ab = generator(L).to(device)

            fake_colorized_images = torch.cat([L, fake_ab], dim=1)
            real_images = torch.cat([L, orig_ab], dim=1)

            # Calculate PSNR for each image in the batch
            batch_psnr = psnr(real_images, fake_colorized_images)
            total_psnr += batch_psnr * L.size(0)
            total_samples += L.size(0)

            # print images
            if batch_idx in rand_batches:
                fake_images_rgb = convert_lab_to_rgb(L, fake_ab)
                real_images_rgb = convert_lab_to_rgb(L, orig_ab)
                plot_images(L[:8], fake_images_rgb[:8], real_images_rgb[:8], batch=batch_idx, image_dir_path=test_dir, mode='test')
                print("\n\n\n")

    avg_psnr = total_psnr / total_samples
    print(f"Average PSNR: {avg_psnr:.2f} dB")