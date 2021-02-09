// $(document).ready(function(){
//
//     var csrf = $("input[name=csrfmiddlewaretoken]").val();
//
//     $('#uploadVideoBtn').on('click','button',function(){
//         $.ajax({
//             url: '/video/',
//             type: 'post',
//             data: {
//                 keyWhatEverYouWant: $(this).text(),
//                 csrfmiddlewaretoken: csrf
//             },
//             success: function (response) {
//                 $('#uploadVideoBtn').text(response.videos)
//             },
//             failure: function(response) {
//                 alert('Got an error dude');
//             }
//         });
//     });
// }

//For checking whether jquery is running or not.
// window.onload = function() {
//     if (window.jQuery) {
//         // jQuery is loaded
//         alert("Yeah!");
//
//     } else {
//         // jQuery is not loaded
//         alert("Doesn't Work");
//     }
// }

$(document).ready(function () {
    $('button#uploadVideoBtn').on('click', function () {
        var myForm = $("form#videoForm");
        if (myForm) {
            if($('#videoUrlId')[0].files.length === 0){
                alert("Please select a video.")
            }else {
                $(this).prop('disabled', true);
                $(myForm).submit();
                $("#loadingView").show();
            }
        }
    });
});


$(document).ready(function(){
    $('button').on('click',function () {
        buttonId = $(this).attr('id');
        console.log(buttonId)
        $('#videoPlayerView').show();
        $('#videoPlayerView source').attr('src', '/media/videos/detections/'+buttonId);
        $("#videoPlayerView")[0].load();
    })
})