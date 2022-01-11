function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#img-area')
                .attr('src', e.target.result)   ; 
            //$('#img-area').css("display", "block")   ;           
        };

        reader.readAsDataURL(input.files[0]);
    }
}

function showCluster() {
    document.getElementById("numofclus").style.display = "block";
}

function hideCluster() {
    document.getElementById("numofclus").style.display = "none";
}