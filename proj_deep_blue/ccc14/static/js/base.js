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

// //For checking whether jquery is running or not.
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
                $("#LoadingText").show();
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
    });
});

function deleteImageByAdmin(id){
    var action = confirm("Do you really wants to delete this image?");
    var endpoint = $("#deleteImageBtn-"+id).attr("data-url");
    var csrf_token = $('input[name=csrfmiddlewaretoken]').val()
    var data = {}

    data['csrfmiddlewaretoken'] = csrf_token;

    if(action != false){
        $.ajax({
            type: "POST",
            url: endpoint,
            data: data,
            dataType: 'json',
            success: function (data) {
                if (data.deleted) {
                  $("#imageRow-" + id).remove();
                }
            },
        });
    }
}

function deleteUserByAdmin(id){
    var action = confirm("Do you really want to delete this user? ");
    var endpoint = $("#deleteUserBtn-"+id).attr("data-url");
    var csrf_token = $("input[name=csrfmiddlewaretoken]").val()
    var data = {}
    data['csrfmiddlewaretoken'] = csrf_token
    if(action != false){
        $.ajax({
            type: "POST",
            url: endpoint,
            data: data,
            dataType: 'Json',
            success: function(data){
                if(data.deleted){
                    $("#deleteTableRow-"+id).remove();
                }else{

                }
            },
        });
    }
}

function addNewUser(form){
    var action = confirm("Do you really want to add this user? ");
    if(action != false){
        $.ajax({
            url: 'add/user/',
            type: "POST",
            data: $(form).serialize(),
            dataType: 'Json',
            success: function(data){
                if(data.created){
                    alert(data.message);
                    $('.modal').modal('toggle');
                    
                }else{
                    alert(data.message);
                    $('.modal').modal('toggle');
                }
            },
        });
    }
}

function updateUser(){
    
}