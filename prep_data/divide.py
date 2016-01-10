import random
if __name__ == '__main__':
    train=open("/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/handshapes/inter/train.txt",'w')
    test=open("/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/handshapes/inter/test.txt",'w')
    crossp=open("/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/handshapes/inter/images.txt")
    for line in crossp:
        if random.random()<0.9:
            train.write(line)
        else:
            test.write(line)