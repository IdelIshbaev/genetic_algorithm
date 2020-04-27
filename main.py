import numpy as np
import cv2 as cv
import random
import sys
from tqdm import tqdm

class Gene:
    def __init__(self, r, g, b, x, y, s):
        self.color_r = r
        self.color_g = g
        self.color_b = b
        self.position_x = x  # from 0 to 512
        self.position_y = y
        self.size = s  # from 10 to 50
    def __lt__(self, other):
        return self.size < other.size
    def copy(self):
        return Gene(self.color_r, self.color_g, self.color_b, self.position_x, self.position_y, self.size)

    color_r = 0
    color_g = 0
    color_b = 0
    position_x = 0 #from 0 to 512
    position_y = 0
    size = 0 #from 10 to 50






def mse(imagea, imageb):
    err = np.sum((imagea.astype("float") - imageb.astype("float")) ** 2)
    err /= float(imagea.shape[0] * imagea.shape[1])
    return err


def make_child(an, fit_rate): #array of chromosomes
    child_ar = []
    for i in range(new_child_amount): #how many chromosomes will be taken to swap genes
        temp_chr_1 = []
        temp_chr_2 = []
        rand_chrom_1 = random.choices(an, fit_rate)
        rand_chrom_2 = random.choices(an, fit_rate)
        for j in range(genes_amount):
            temp_gene_1 = Gene(0, 0, 0, 0, 0, 0)
            temp_gene_1 = rand_chrom_1[0][j].copy()
            temp_chr_1.append(temp_gene_1)
            temp_gene_2 = Gene(0, 0, 0, 0, 0, 0)
            temp_gene_2 = rand_chrom_2[0][j].copy()
            temp_chr_2.append(temp_gene_2)

        for k in range(len(temp_chr_1)):
            r = random.randint(0,10)
            if r <= prob_of_change:
                temp_gene_0 = temp_chr_1[k].copy()
                temp_chr_1[k] = temp_chr_2[k].copy()
                temp_chr_2[k] = temp_gene_0.copy()
        child_ar.append(temp_chr_1) #add to list of chromosomes
        child_ar.append(temp_chr_2) #add to list of chromosomes
    # return child_ar
    a = child_ar.copy()
    for i in range(len(a)):
        for j in range(genes_amount): #index of genes
            p1 = random.randint(0, 10)
            if p1 <= prob_of_change:
                p2 = random.randint(0, 10)
                if p2 <= prob_of_change:
                    a[i][j].color_r = random.randint(0,225)
                p2 = random.randint(0, 10)
                if p2 <= prob_of_change:
                    a[i][j].color_g = random.randint(0,225)
                p2 = random.randint(0, 10)
                if p2 <= prob_of_change:
                    a[i][j].color_b = random.randint(0,225)
                p2 = random.randint(0, 10)
                if p2 <= prob_of_change:
                    a[i][j].position_x = random.randint(0,pixs-max_size)
                p2 = random.randint(0, 10)
                if p2 <= prob_of_change:
                    a[i][j].position_y = random.randint(0,pixs-max_size)
                p2 = random.randint(0, 10)
                if p2 <= prob_of_change:
                        a[i][j].size = random.randint(2,max_size)
    return a


# def change_prop(in_a): #array of child chromosomes
#
#     return a

def get_fit_rate(fit_img, fit_chrom):
    fit_fit_rate = []
    for i in range(len(fit_chrom)):
        fit_fit_rate.append(mse(fit_img, fit_chrom[i]))
    return fit_fit_rate


def creare_image(chromosomes):
    cr_chromosomes_img = []
    for i in range(len(chromosomes)):
        img = np.zeros([pixs, pixs, 3], dtype=np.uint8)
        img.fill(255)
        for j in range(genes_amount):
            cv.rectangle(img, (chromosomes[i][j].position_x, chromosomes[i][j].position_y),
                         (chromosomes[i][j].position_x+chromosomes[i][j].size, chromosomes[i][j].position_y+chromosomes[i][j].size), (chromosomes[i][j].color_r,chromosomes[i][j].color_g, chromosomes[i][j].color_b), -1)
        cr_chromosomes_img.append(img)
    return cr_chromosomes_img


def run_algo(input_image):
    chromosomes = []
    output_chromosomes_img = []

    # create population
    for i in range(chromosomes_amount):
        temp_chromosomes = []
        for j in range(genes_amount):
            temp_Gene = Gene(random.randint(255,255), random.randint(255,255), random.randint(255,255), random.randint(0,pixs-max_size),random.randint(0,pixs-max_size), random.randint(2,max_size))
            temp_chromosomes.append(temp_Gene)
        chromosomes.append(temp_chromosomes)

    fit_rate_check = 80000
    ki = -1
    repeat_amount = 0
    prev_fit_rate = 80000
    # for k in tqdm(range(iter)):
    for k in range(iter):
    # while fit_rate_check > 30000:
        ki += 1
        chromosomes_img = creare_image(chromosomes)    # create first images

        fit_rate = get_fit_rate(input_image, chromosomes_img)    # make fittest and sort

        chromosomes1 = [x for _, x in sorted(zip(fit_rate, chromosomes))] #sort
        chromosomes = chromosomes1
        fit_rate.sort()
        child_chromosomes = make_child(chromosomes, fit_rate)    #make mutation
        chromosomes = chromosomes + child_chromosomes

        if k == 0:     # very first image
            cv.imshow('kakakha_kotoraya_vonyaet', chromosomes_img[0])

        chromosomes_img = creare_image(chromosomes)    #make new images

        fit_rate = get_fit_rate(input_image, chromosomes_img)    # make fittest

        chromosomes1 = [x for _, x in sorted(zip(fit_rate, chromosomes))] #sort
        chromosomes = chromosomes1

        for i in range(new_child_amount*2):
             chromosomes.pop()
        if k == iter-1:
            output_chromosomes_img.append(chromosomes_img[0])

        fit_rate.sort()
        for i in range(len(fit_rate)):
            fit_rate[i] = int(fit_rate[i])
        print(fit_rate[0], ki, repeat_amount)
        if fit_rate[0] == prev_fit_rate:
            repeat_amount += 1
        else:
            repeat_amount = 0
        if repeat_amount >= 10:
            # print("change last 2")
            for i in range(5):
                chromosomes.pop()
            # create population
            for i in range(5):
                temp_chromosomes_0 = []
                for j in range(genes_amount):
                    temp_Gene = Gene(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255),
                                     random.randint(0, pixs - max_size), random.randint(0, pixs - max_size),
                                     random.randint(2, max_size))
                    temp_chromosomes_0.append(temp_Gene)
                chromosomes.append(temp_chromosomes_0)
        prev_fit_rate = fit_rate[0]

        fit_rate_check = fit_rate[0]
        return output_chromosomes_img[0]

chromosomes_amount = 15
genes_amount = 100
new_child_amount = 6 # n*2
iter = 3

prob_of_change = 8
max_size = 10
divider = 2
pixs = int(225/divider)

#
# cv.imshow('iskustvo',input_image)
# cv.imshow('kakakha',output_chromosomes_img[0])
#
# cv.waitKey(0)
# cv.destroyAllWindows()