

if __name__ == '__main__':
    with open("standard.sbn.drs.out", "r", encoding="utf-8") as f:
        with open("Neuralboxer_en_standard_dev.txt", "w", encoding="utf-8") as w:
            data = f.readlines()
            for i in range(len(data)):
                if i % 2 == 0:
                    w.write(data[i])