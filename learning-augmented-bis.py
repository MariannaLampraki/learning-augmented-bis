import json
import time
import math
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor() # επιλέξαμε το μοντέλο oracle Decision Tree Regressor διότι ήταν πιο αποδωτικό στις δοκιμές μας
# ------------------ Βοηθητικες Συναρτησεις ------------------

# Η συνάρτηση αυτή παίρνει γραμμικά δεδομένα, όπως ο χρόνος, και τα μετατρέπει σε τριγωνομετρικά δεδομένα, διότι όπως εξηγήθηκε
# και στην αναφορά, οι μήνες, μέρες, ώρες και λεπτά δεν έχουν γραμμική σχέση μεταξύ τους, δηλαδή, για παράδειγμα, οι ώρες 23:30
# με τη 00:00 για εμάς είναι πολύ κοντά αλλά για ένα μοντέλο μηχανικής μάθησης που αντιμετωπίζει τις ώρες και τα λεπτά ως
# κάτι γραμμικό, αυτές οι δύο ώρες έχουν πολύ μεγάλη απόσταση. Ωστόσο, με τη συνάρτηση αυτή τονίζεται η χρονική αυτή σχέση.
def cycle_timestamp(test_month, test_day, test_hours, test_minutes):
    pi = math.pi
    hour_sin = math.sin(2 * pi * test_hours / 24)
    hour_cos = math.cos(2 * pi * test_hours / 24)
    minute_sin = math.sin(2 * pi * test_minutes / 60)
    minute_cos = math.cos(2 * pi * test_minutes / 60)
    day_sin = math.sin(2 * pi * test_day / 31)
    day_cos = math.cos(2 * pi * test_day / 31)
    month_sin = math.sin(2 * pi * test_month / 12)
    month_cos = math.cos(2 * pi * test_month / 12)
    return [month_cos, month_sin, day_sin, day_cos, minute_sin, minute_cos, hour_sin, hour_cos]

def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        day_data = json.loads(line.strip())
        for timestamp, value in day_data.items():
            try:
                data.append((timestamp, float(value)))
            except ValueError:
                continue
    data = sorted(data, key=lambda x:x[0]) # δεν επιστρέφουμε τη διατεταγμένη λίστα ακόμα

    # τα sets για την εκπαίδευση του μοντέλου μας
    features = [] # αυτό είναι το set με τα δεδομένα που εισάγουμε στο μοντέλο μας
    targets = [] # αυτό είναι το set με τα δεδομένα που περιμένουμε το μοντέλο μας να μπορεί να προβλέψει

    # ο λόγος που χρησιμοποιούμε νέο βρόχο for για αποθήκευση των sets είναι επειδή έπειτα κάνουμε sort την data άρα
    # τα targets (όπως βλέπουμε πιο κάτω) δε θα αντιστοιχούσαν στις σωστές τιμές των timestamps
    for i, (timestamp,_) in enumerate(data):
        month = int(timestamp[5:7])
        day = int(timestamp[8:10])
        hour = int(timestamp[11:13])
        minute = int(timestamp[14:16])
        # το features περιέχει τις τριγωνομετρικές τιμές των timestamps
        # δεν περιλαμβάνουμε το έτος επειδή όλα τα timestamps στα δεδομένα μας είναι από το 2014, διαφορετικά θα το περιλαμβάναμε
        features.append(cycle_timestamp(month, day, hour, minute))
        # το targets περιλαμβάνει τη θέση του timestamp μέσα στη νέα, διατεταγμένη δομή
        targets.append(i)

    # ο λόγος που η μεταβλητή model ορίζεται έξω από τη συνάρτηση και χρησιμοποιείτα εδώ με global είναι επειδή θα το
    # χρεαιστούμε και στη συνάρτηση labis() άρα πρέπει να είναι καθολικά ορισμένο
    global model
    model.fit(features, targets) # εκπαιδεύουμε το μοντέλο με όλα τα δεδομένα μας
    # όπως αναφέραμε και στην αναφορά, δε χωρίσαμε τα features και targets σε training και testing sets επειδή:
    # α) δε μας ζητείται από την άσκηση να αξιολογήσουμε το μοντέλο μας
    # β) η αξιολόγηση του μοντέλου έχει γίνει ήδη από εμάς όταν επιλέγαμε το καταλληλότερο μοντέλο
    # γ) αφού δεν είναι απαραίτητα τα testing sets, θεωρήσαμε καλύτερη και αποδοτικότερη λύση να μη δεσμεύουμε περισσότερο χώρο στη μνήμη και να εκπαιδεύσουμε το μοντέλο με όσο περισσότερα δεδομένα γίνεται
    # αν θέλαμε να τα χωρίσουμε, θα κάναμε χρήση των train_test_split και sklearn.metrics για να χωρίσουμε τα sets και να τα αξιολογίσουμε με στατιστικά στοιχεία

    return data

def timetoint(timestamp):
    return (int(timestamp[0:4]) * 10000000000 +
            int(timestamp[5:7]) * 100000000 +
            int(timestamp[8:10]) * 1000000 +
            int(timestamp[11:13]) * 10000 +
            int(timestamp[14:16]) * 100 +
            int(timestamp[17:19]))

def binarySearch(data, left, right, key):
    while left <= right:
        mid = (left + right) // 2
        mid_val = timetoint(data[mid][0])

        if mid_val == key:
            return data[mid]
        elif mid_val < key:
            return binarySearch(data, mid+1, right, key)
        else:
            return binarySearch(data, left, mid-1, key)
    return None

bis_count = 0 # Η μεταβλητή αυτή χρησιμοποιείται για να μετρήσουμε πόσα "βήματα" κάνει ο BIS μέχρι να βρει τη σωστή τιμή
def bis(data, target):
    global bis_count
    bis_count = 0 # αρχικοποίηση της bis_count
    x = timetoint(target)
    left, right = 0, len(data) - 1

    while left <= right:
        size = right - left + 1

        S_left = timetoint(data[left][0])
        S_right = timetoint(data[right][0])

        if S_right == S_left:
            break

        next_el = left + math.ceil(size * (x - S_left) / (S_right - S_left))
        next_el = min(max(left, next_el), right)

        S_next = timetoint(data[next_el][0])
        if x == S_next:
            return data[next_el]

        i = 1
        jump = int(math.sqrt(size))

        if x > S_next:
            while True:
                bis_count += 1
                idx = next_el + i * jump - 1
                if idx > right or x <= timetoint(data[idx][0]):
                    break
                i += 1
            left = min(next_el + (i - 1) * jump, right)
            right = min(next_el + i * jump, right)

        else:
            while True:
                bis_count += 1
                idx = next_el - i * jump + 1
                if idx < left or x >= timetoint(data[idx][0]):
                    break
                i += 1
            right = max(next_el - (i - 1) * jump, left)
            left = max(next_el - i * jump, left)
    return None


# ----------------------------------- LEARNING AUGMENTED BIS --------------------------------------------
labis_count = 0 # μετράμε πόσα βήματα κάνει ο labis μέχρι να βρει τη σωστή τιμή
def labis(data, target):
    global labis_count
    labis_count = 0 # αρχικοποίηση της labis
    x = timetoint(target)
    left, right = 0, len(data) - 1

    while left <= right:
        size = right - left + 1
        S_left = timetoint(data[left][0])
        S_right = timetoint(data[right][0])

        if S_right == S_left:
            break

        # Βρίσκουμε τις τιμές του μήνα, ημέρας, ώρας και λεπτού
        test_month = (x//100000000)%100
        test_day = (x//1000000)%100
        test_hours = (x//10000)%100
        test_minutes = (x//100)%100
        # τις μετατρέπουμε σε τριγωνομετρικούς αριθμούς
        test_input = [cycle_timestamp(test_month, test_day, test_hours, test_minutes)]
        # χρησιμοποιούμε πρόβλεψη για το πρώτο στοιχείο που θα ελέγξουμε
        next_el = math.ceil(model.predict(test_input)[0])
        next_el = min(max(left, next_el), right)

        S_next = timetoint(data[next_el][0])
        if x == S_next:
            return data[next_el]

        i = 1
        jump = int(math.sqrt(size))

        if x > S_next:
            while True:
                labis_count += 1
                idx = next_el + i * jump - 1
                if idx > right or x <= timetoint(data[idx][0]):
                    break
                i += 1
            left = min(next_el + (i - 1) * jump, right)
            right = min(next_el + i * jump, right)

        else:
            while True:
                labis_count += 1
                idx = next_el - i * jump + 1
                if idx < left or x >= timetoint(data[idx][0]):
                    break
                i += 1
            right = max(next_el - (i - 1) * jump, left)
            left = max(next_el - i * jump, left)
    return None


bis_star_count = 0 # με αυτή τη μεταβλητή μετράμε τα "βήματα" που θα χρειαστεί ο BIS* μέχρι να βρεί το σωστό αποτέλεσμα
def bis_star(data, target):
    global bis_star_count
    bis_star_count = 0 # αρχικοποίηση της bis_star_count
    x = timetoint(target)
    left, right = 0, len(data) - 1
    while left <= right:
        size = right - left + 1
        S_left = timetoint(data[left][0])
        S_right = timetoint(data[right][0])

        if S_right == S_left:
            break

        next_el = left + math.ceil(size * (x - S_left) / (S_right - S_left))
        next_el = min(max(left, next_el), right)

        S_next = timetoint(data[next_el][0])
        if x == S_next:
            return data[next_el]
        i = 1
        jump = int(math.sqrt(size))

        if x > S_next:

            prev_jump_idx = next_el
            while True:
                bis_star_count += 1
                idx = next_el + i * jump - 1
                if idx > right or x <= timetoint(data[idx][0]):
                    break
                prev_jump_idx = idx
                i *= 2

            bin_left = prev_jump_idx
            bin_right = min(next_el + i * jump, right)
            return binarySearch(data, bin_left, bin_right, x)

        else:
            prev_jump_idx = next_el
            while True:
                bis_star_count += 1
                idx = next_el - i * jump + 1
                if idx < left or x >= timetoint(data[idx][0]):
                    break
                prev_jump_idx = idx
                i *= 2
            bin_right = prev_jump_idx
            bin_left = max(next_el - i * jump, left)
            return binarySearch(data, bin_left, bin_right, x)
    return None

# ------------------ Main ------------------
def main():
    temp_data = load_data('tempm.txt')
    hum_data = load_data('hum.txt')

    print("Επιλογή τύπου δεδομένων:")
    print("1. Θερμοκρασία")
    print("2. Υγρασία")
    print("3. Και τα δύο")
    print("4. Εξοδος")
    while True: # επανάλαβε όσο ο χρήστης δεν επιλέγει "Έξοδος"
        choice = input("Δώσε επιλογή (1/2/3/4): ")

        # εδώ ορίζονται οι μεταβλητές μέτρησης βημάτων ως οι καθολικές μεταβλητές που αρχικοποιήθηκαν μέσα στις αντίστοιχες συναρτήσεις
        global bis_count
        global labis_count
        global bis_star_count
        while choice != "1" and choice != "2" and choice != "3" and choice != "4": # έλεγχος έγκυρης επιλογής
            print("Η επιλογη δεν υπάρχει!")
            choice = input("Δώσε επιλογή (1/2/3/4): ")

        if choice != "4":
            timestamp = input("Δώσε timestamp (π.χ. 2014-03-21T214:00:00): ").replace(' ', 'T')

        # Θερμοκρασία/temp
        if choice == "1" or choice == "3":

            start = time.time()
            result_bis = bis(temp_data, timestamp)
            time_bis = time.time() - start

            start = time.time()
            result_bis_star = bis_star(temp_data, timestamp)
            time_bis_star = time.time() - start

            start = time.time()
            result_labis = labis(temp_data, timestamp)
            time_labis = time.time() - start

            print("\nΘερμοκρασία:")
            if result_bis:
                print(f"[BIS] Βρέθηκε: {result_bis[1]} °C (σε {time_bis:.6f} sec)")
                print(f"[BIS] Εκτελέστηκαν: {bis_count} επαναλήψεις.\n")
            else:
                print("[BIS] Δεν βρέθηκε θερμοκρασία για αυτό το timestamp.")
            if result_bis_star:
                print(f"[BIS*] Βρέθηκε: {result_bis_star[1]} °C (σε {time_bis_star:.6f} sec)")
                print(f"[BIS*] Εκτελέστηκαν: {bis_star_count} επαναλήψεις.\n")
            else:
                print("[BIS*] Δεν βρέθηκε θερμοκρασία για αυτό το timestamp.")
            if result_labis:
                print(f"[LABIS] Βρεθηκε: {result_labis[1]} °C (σε {time_labis:.6f} sec)")
                print(f"[LABIS] Εκτελέστηκαν {labis_count} επαναλήψεις.\n")
            else:
                print(f"[LABIS] Δεν βρέθηκε θερμοκρασία για αυτό το timestamp.")

        # Υγρασία/hum
        if choice == "2" or choice == "3":
            start = time.time()
            result_bis = bis(hum_data, timestamp)
            time_bis = time.time() - start

            start = time.time()
            result_bis_star = bis_star(hum_data, timestamp)
            time_bis_star = time.time() - start

            start = time.time()
            result_labis = labis(hum_data, timestamp)
            time_labis = time.time() - start

            print("\nΥγρασία:")
            if result_bis:
                print(f"[BIS] Βρέθηκε: {result_bis[1]} % (σε {time_bis:.6f} sec)")
                print(f"[BIS] Εκτελέστηκαν: {bis_count} επαναλήψεις.\n")
            else:
                print("[BIS] Δεν βρέθηκε υγρασία για αυτό το timestamp.")
            if result_bis_star:
                print(f"[BIS*] Βρέθηκε: {result_bis_star[1]} % (σε {time_bis_star:.6f} sec)")
                print(f"[BIS*] Εκτελέστηκαν: {bis_star_count} επαναλήψεις.\n")
            else:
                print("[BIS*] Δεν βρέθηκε υγρασία για αυτό το timestamp.")
            if result_labis:
                print(f"[LABIS] Βρεθηκε: {result_labis[1]} % (σε {time_labis:.6f} sec)")
                print(f"[LABIS] Εκτελέστηκαν {labis_count} επαναλήψεις.\n")
            else:
                print("[LABIS] Δεν βρέθηκε υγρασία για αυτό το timestamp.")

        if choice == "4":
            break

if __name__ == "__main__":
    main()