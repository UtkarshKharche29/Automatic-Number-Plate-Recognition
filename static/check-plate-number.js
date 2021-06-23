let plate_nos = [];
var displayMsg = "Not Valid";

function submit() {
  fetch("https://jsonplaceholder.typicode.com/users") //dummy data
    .then((response) => response.json())
    .then((data) => {
      const plate_no = document.getElementById("inputPlateNo").value;

      plate_nos = data.map((value) => {
        return value.name;
      });

      console.log(plate_nos);

      for (let i = 0; i < plate_nos.length; i++) {
        if (plate_nos[i] == plate_no) {
          displayMsg = "Valid";
          return displayMsg;
        }
      }
    })
    .then(
      (displayMsg) =>
        (document.getElementById("display").innerHTML =
          displayMsg || "Not Valid")
    )
    .catch((err) => console.log("Request Failed", err));
}
