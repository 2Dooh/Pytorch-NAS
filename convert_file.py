import os
import re

def convert_file(path, pdf_to_jpg=False):
    # path = './experiments/exp-20210323-1754-20210324T024238Z-001/exp-20210323-1754/checkpoints/gifs/FLOPS_NTK'
    files = os.listdir(path)
    func = pdf2jpg if pdf_to_jpg else rename
    for i, file in enumerate(files):
        # idx = re.search(r'(\d+)', file).group(1)
        # command = 'convert -density 150 {} -quality 100 {}.jpg'.format(
        #     os.path.join(path, file), os.path.join(path, idx))
        # os.system(command)
        func(path, file)

        # idx = re.search(r'_(\d+)', file).group(1)
        # os.rename(
        #     os.path.join(path, file), 
        #     os.path.join(path, '{}.pdf'.format(idx)))

def pdf2jpg(path, file):
    idx = re.search(r'(\d+)', file).group(1)
    command = 'convert -density 150 {} -quality 100 {}.jpg'.format(
        os.path.join(path, file), os.path.join(path, idx))
    os.system(command)

def rename(path, file):
    idx = re.search(r'_(\d+)', file).group(1)
    os.rename(
        os.path.join(path, file), 
        os.path.join(path, '{}.pdf'.format(idx)))


path = './experiments/exp-20210324-1853/checkpoints/gifs'
convert_file(path, pdf_to_jpg=True)

# # path = './experiments/exp-20210323-1754-20210324T024238Z-001/exp-20210323-1754/checkpoints/gifs/FLOPS_NTK'
# files = os.listdir(path)
# # func = pdf2jpg if pdf_to_jpg else rename
# for i, file in enumerate(files):
#     # idx = re.search(r'(\d+)', file).group(1)
#     # command = 'convert -density 150 {} -quality 100 {}.jpg'.format(
#     #     os.path.join(path, file), os.path.join(path, idx))
#     # os.system(command)
#     # func(path, file)

#     # idx = re.search(r'_(\d+)', file).group(1)
#     # os.rename(
#     #     os.path.join(path, file), 
#     #     os.path.join(path, '{}.pdf'.format(idx)))

