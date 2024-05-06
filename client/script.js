const btnImagesUpload = document.getElementById("btnImagesUpload");
const imageInput = document.getElementById("imageInput");
const dropBox = document.querySelector(".drop-box");
const imageBox = document.querySelector(".image-container");
const dropdownMenu = document.querySelector(".dropdown-menu");
const dropdownBtn = document.getElementById("dropdownImageSelector");
const imageDisplay = document.querySelector("img");
const btnClassificar = document.getElementById("btnClassificar");
const lblImgClass = document.getElementById("imgClass");
const lblImgClassPctg = document.getElementById("imgClassPctg");

var selectedImage = null;
var selectedFiles = [];

const reader = new FileReader();
reader.onload = () => {
  imageDisplay.src = reader.result;
};

dropBox.addEventListener("dragover", (e) => {
  e.preventDefault();
})

dropBox.addEventListener("drop", (e) => {
  e.preventDefault();

  const files = e.dataTransfer.files;
  handleImages(files);
});

btnImagesUpload.onclick = () => {
  imageInput.click();
};

imageInput.onchange = () => {
  handleImages(imageInput.files);
};

btnClassificar.onclick = () => {
  if (selectedImage) {
    const formData = new FormData();
    formData.append('file', selectedImage);

    fetch('http://127.0.0.1:5000/classificar', {
      method: 'POST',
      body: formData
    })
    .then(response => {
      if (!response.ok){
        alert("Erro ao classificar imagem");
      }
      return response.json();
    })
    .then(data => {
      lblImgClass.innerText = "Classificação: " + data["classificacao"];
      lblImgClassPctg.innerText = "Certeza: " + data["pctg"] + "%";
    })
    .catch(error => {
      alert("Erro ao classificar imagem. Verifique se o servidor Flask está funcionando.");
    })
  }
};

function setSelectedImage(image) {
  lblImgClass.innerText = "";
  lblImgClassPctg.innerText = "";
  dropdownBtn.innerText = image.name;
  reader.readAsDataURL(image);
  selectedImage = image;
}

function handleImages(images) {
  selectedFiles = images;
  dropBox.classList.add("vanish");

  setTimeout(() => {
    dropBox.classList.add("d-none");
    imageBox.classList.remove("d-none", "vanish");
  }, 1000);

  let image = selectedFiles[0];
  dropdownBtn.innerText = image.name;
  reader.readAsDataURL(image);
  selectedImage = image;

  for(let i=0; i < selectedFiles.length; i++){
    const item = document.createElement("a");
    const file = selectedFiles[i];

    item.className = "dropdown-item text-white"
    item.innerText = file.name;
    item.onclick = () => setSelectedImage(file);

    dropdownMenu.appendChild(item);
  }
}