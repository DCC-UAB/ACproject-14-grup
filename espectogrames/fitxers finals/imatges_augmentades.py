import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def augment_image(image):
    """
    Aplica augmentació a una imatge.
    """
    flipped = np.fliplr(image)
    noisy = np.clip(image + np.random.normal(0, 0.02, image.shape), 0, 1)
    brighter = np.clip(image * 1.2, 0, 1)
    darker = np.clip(image * 0.8, 0, 1)
    return [image, flipped, noisy, brighter, darker]

def mostrar_i_guardar_augmentacio(img_path, output_dir="output_augmentacio"):
    """
    Llegeix una imatge, aplica augmentació, la visualitza i la guarda en un directori.
    """
    # Crear directori per guardar les imatges augmentades
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Llegir i normalitzar la imatge
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("[ERROR] No s'ha pogut carregar la imatge.")
        return

    img_resized = cv2.resize(img, (128, 128)) / 255.0
    augmented_images = augment_image(img_resized)

    # Visualitzar i guardar la imatge original i augmentades
    titols = ["Original", "Flipped", "Noisy", "Brighter", "Darker"]
    plt.figure(figsize=(12, 6))

    for i, (aug_img, titol) in enumerate(zip(augmented_images, titols)):
        plt.subplot(1, 5, i + 1)
        plt.imshow(aug_img, cmap="gray")
        plt.title(titol)
        plt.axis("off")

        # Guardar imatge augmentada
        output_path = os.path.join(output_dir, f"{titol.lower()}.png")
        cv2.imwrite(output_path, (aug_img * 255).astype(np.uint8))

    plt.tight_layout()
    plt.show()
    print(f"[SUCCESS] Imatges augmentades guardades a '{output_dir}'")

if __name__ == "__main__":
    # Exemple: Ruta de la imatge original
    img_path = "ACproject-14-grup/datasets/Data1/images_original/blues/blues00000.png"
    print("[INFO] Aplicant augmentació a una imatge per la presentació...")
    mostrar_i_guardar_augmentacio(img_path)
