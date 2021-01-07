var btnContainer = document.getElementById("baseList");

// Get all buttons with class="btn" inside the container
var btns = btnContainer.getElementsByClassName("navBaseList");

// Loop through the buttons and add the active class to the current/clicked button
for (var i = 0; i < btns.length; i++) {
    btns[i].addEventListener("click", function() {
        var current = document.getElementsByClassName("active");
        current[0].className = current[0].className.replace(" active", "");
        this.className += " active";
    });
}

$('input[type="file"]').on('change', function(e){
    var fileName = e.target.files[0].name;
    if (fileName) {
        $(e.target).parent().attr('data-message', fileName);
    }
});
