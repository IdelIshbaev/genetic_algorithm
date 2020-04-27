import numpy as np
import cv2 as cv
import random
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

#calculate fit rate of input image and generated image
def mse(imagea, imageb):
    err = np.sum((imagea.astype("float") - imageb.astype("float")) ** 2)
    err /= float(imagea.shape[0] * imagea.shape[1])
    return err

#merge genes of some chromosomes and get new one
def make_child(an, fit_rate): #array of chromosomes
    child_ar = []
    for i in range(new_child_amount): #how many chromosomes will be taken to swap genes
        temp_chr_1 = []
        temp_chr_2 = []
        rand_chrom_1 = random.choices(an, fit_rate) #take random chromosome from best ones
        rq = random.randint(0,1)
        rand_chrom_2 = random.choices(an, reversed(fit_rate)) #rake random chromosome from worst ones

        for j in range(genes_amount): #cope genes of chosen chromosomes to new ones
            temp_gene_1 = [0, 0, 0, 0, 0, 0]
            temp_gene_1 = rand_chrom_1[0][j].copy()
            temp_chr_1.append(temp_gene_1)
            temp_gene_2 = [0, 0, 0, 0, 0, 0]
            temp_gene_2 = rand_chrom_2[0][j].copy()
            temp_chr_2.append(temp_gene_2)

        for k in range(len(temp_chr_1)):  #swap chromosomes randomly
            r = random.randint(0,10)
            if r <= prob_of_change:
                temp_gene_0 = temp_chr_1[k].copy()
                temp_chr_1[k] = temp_chr_2[k].copy()
                temp_chr_2[k] = temp_gene_0.copy()
        child_ar.append(temp_chr_1) #add to list of chromosomes
        child_ar.append(temp_chr_2) #add to list of chromosomes
    # return child_ar
    a = child_ar.copy() #mutation, change genes of new chromosomes to random ones
    for i in range(len(a)):
        for j in range(genes_amount): #index of genes
            p1 = random.randint(0, 10)
            if p1 <= prob_of_change:
                p2 = random.randint(0, 10)
                t_size = random.randint(min_size, max_size)
                if p2+2 <= prob_of_change:
                    # a[i][j][0] = random.randint(0,225)
                    rand_pl = random.randint(0,1)
                    # if rand_pl == 1:
                    #     rand_col = random.randint(0, 255 - a[i][j][0])
                    #     a[i][j][0] = a[i][j][0]+rand_col
                    # else:
                    #     rand_col = random.randint(0, a[i][j][0])
                    #     a[i][j][0] = a[i][j][0]-rand_col
                    if rand_pl == 1:
                        a[i][j][0] = a[i][j][0]+10
                        if a[i][j][0] > 255:
                            a[i][j][0] = 255
                    else:
                        a[i][j][0] = a[i][j][0]-10
                        if a[i][j][0] < 0:
                            a[i][j][0] = 0
                p2 = random.randint(0, 10)
                if p2+2 <= prob_of_change:
                    # a[i][j][1] = random.randint(0,225)
                    rand_pl = random.randint(0,1)
                    # if rand_pl == 1:
                    #     rand_col = random.randint(0, 255 - a[i][j][1])
                    #     a[i][j][1] = a[i][j][1]+rand_col
                    # else:
                    #     rand_col = random.randint(0, a[i][j][1])
                    #     a[i][j][1] = a[i][j][1]-rand_col
                    if rand_pl == 1:
                        a[i][j][1] = a[i][j][1]+10
                        if a[i][j][1] > 255:
                            a[i][j][1] = 255
                    else:
                        a[i][j][1] = a[i][j][1]-10
                        if a[i][j][1] < 0:
                            a[i][j][1] = 0
                p2 = random.randint(0, 10)
                if p2+2 <= prob_of_change:
                    # a[i][j][2] = random.randint(0,225)
                    rand_pl = random.randint(0,1)
                    # if rand_pl == 1:
                    #     rand_col = random.randint(0, 255 - a[i][j][2])
                    #     a[i][j][2] = a[i][j][2]+rand_col
                    # else:
                    #     rand_col = random.randint(0, a[i][j][2])
                    #     a[i][j][2] = a[i][j][2]-rand_col
                    if rand_pl == 1:
                        a[i][j][2] = a[i][j][2]+10
                        if a[i][j][2] > 255:
                            a[i][j][2] = 255
                    else:
                        a[i][j][2] = a[i][j][2]-10
                        if a[i][j][2] < 0:
                            a[i][j][2] = 0
                p2 = random.randint(0, 10)
                if p2 <= prob_of_change:
                    # a[i][j][3] = random.randint(0,pixs-t_size)
                    a[i][j][3] = a[i][j][3]
                p2 = random.randint(0, 10)
                if p2 <= prob_of_change:
                    # a[i][j][4] = random.randint(0,pixs-t_size)
                    a[i][j][4] = a[i][j][4]
                p2 = random.randint(0, 10)
                if p2 <= prob_of_change:
                    a[i][j][5] = t_size
    return a #return new chromosomes

#get fit rate list of generated images and input image
def get_fit_rate(fit_img, fit_chrom):
    fit_fit_rate = []
    for i in range(len(fit_chrom)):
        fit_fit_rate.append(mse(fit_img, fit_chrom[i]))
    return fit_fit_rate #return list of fit rates


def creare_image(chromosomes): #generate images from chromosomes
    cr_chromosomes_img = []
    for i in range(len(chromosomes)):
        img = np.zeros([pixs, pixs, 3], dtype=np.uint8)
        img.fill(255)
        for j in range(genes_amount):
            cv.rectangle(img, (chromosomes[i][j][3], chromosomes[i][j][4]),
                         (chromosomes[i][j][3]+chromosomes[i][j][5], chromosomes[i][j][4]+chromosomes[i][j][5]), (chromosomes[i][j][0],chromosomes[i][j][1], chromosomes[i][j][2]), -1)
        cr_chromosomes_img.append(img)
    return cr_chromosomes_img #return images


chromosomes_amount = 150
genes_amount = 100
new_child_amount = 90 # n*2
iter = 1000000000

prob_of_change = 5
max_size = 20
min_size = 10
pixs = 225

chromosomes = []
output_chromosomes_img = []

#read input image
path = r'/Users/idel_isbaev/PycharmProjects/IAI_ass/genetic_algorithm/images/img_04.png'
input_image = cv.imread(path)

#print input image as plot
plt.subplot(121)
plt.axis("off")
plt.imshow(cv.cvtColor(input_image, cv.COLOR_BGR2RGB))
plt.subplot(122)
plt.axis("off")
plt.imshow(cv.cvtColor(input_image, cv.COLOR_BGR2RGB))
plt.show()

if input_image is None:
    sys.exit("Could not read the image.")
# create population
for i in range(chromosomes_amount):
    temp_chromosomes = []
    for j in range(genes_amount):
        t_size = random.randint(min_size, max_size)
        temp_Gene = [random.randint(0,255), random.randint(0,255), random.randint(0,255), random.randint(0,pixs-t_size),random.randint(0,pixs-t_size), t_size]
        temp_chromosomes.append(temp_Gene)
    chromosomes.append(temp_chromosomes)

fit_rate_check = 100000
ki = -1
repeat_amount = 0
prev_fit_rate = 80000
for k in tqdm(range(iter)): #make loop for iterations
# for k in range(iter):
# while fit_rate_check > 1000:
    ki += 1
    chromosomes_img = creare_image(chromosomes)    # create first images
    fit_rate = get_fit_rate(input_image, chromosomes_img)    # make fittest and sort
    output_fit = fit_rate.copy()

    chromosomes_child = make_child(chromosomes, fit_rate)    #make mutation
    chromosomes = chromosomes + chromosomes_child

    chromosomes_img_child = creare_image(chromosomes_child)    #make new images
    chromosomes_img = chromosomes_img + chromosomes_img_child
    fit_rate_child = get_fit_rate(input_image, chromosomes_img_child)    # make fittest
    fit_rate = fit_rate + fit_rate_child
    output_fit1 = fit_rate_child.copy()
    output_fit = output_fit + output_fit1

    for i in range(new_child_amount*2): #delete not needed chromosomes (with worst fit rate)
        ind_t = fit_rate.index(max(fit_rate))
        fit_rate.pop(ind_t)
        chromosomes.pop(ind_t)
        chromosomes_img.pop(ind_t)

    min_fit_rate = min(fit_rate)
    if min_fit_rate == prev_fit_rate: #monitor how many iterations were made without changing best one
        repeat_amount += 1
    else:
        repeat_amount = 0
    prev_fit_rate = min_fit_rate
    fit_rate_check = min_fit_rate
    if repeat_amount == 0 or fit_rate_check == 1000:
        output_ind_t = fit_rate.index(min_fit_rate)

    if repeat_amount >= 10:  #if we dont have chages for 10 iterations, we change worst half of chromosomes to random ones
        for i in range(int(chromosomes_amount/2)): #delete
            ind_t = fit_rate.index(max(fit_rate))
            fit_rate.pop(ind_t)
            chromosomes.pop(ind_t)
        # create population
        for i in range(int(chromosomes_amount/2)): #generate new ones
            temp_chromosomes_0 = []
            temp_chromosomes = []
            for j in range(genes_amount):
                t_size = random.randint(min_size, max_size)
                temp_Gene = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255),
                             random.randint(0, pixs - t_size), random.randint(0, pixs - t_size), t_size]
                temp_chromosomes_0.append(temp_Gene)
            chromosomes.append(temp_chromosomes_0)

    if repeat_amount == 0 or fit_rate_check == 1000: #print if best image was changed
        print(mse(chromosomes_img[output_ind_t], input_image))
        plt.subplot(121)
        plt.axis("off")
        plt.imshow(cv.cvtColor(input_image, cv.COLOR_BGR2RGB))
        plt.subplot(122)
        plt.axis("off")
        plt.imshow(cv.cvtColor(chromosomes_img[output_ind_t], cv.COLOR_BGR2RGB))
        plt.show()


cv.waitKey(0)
cv.destroyAllWindows()