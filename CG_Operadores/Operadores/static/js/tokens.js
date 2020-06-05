$(document).ready(function(){

    $('#algoritmo').on('change',function(){
        var valor = $(this).val();

        switch (valor){
            case "1":
                $('#bot').children('#const1').show();
                $('#bot').children('#const2').show();
                $('#bot').children('#const3').hide();
                $('#bot').children('#const4').hide();
                $('#constante').val(127);
                $('#constante1').val(255);
                $('#resul').children('#enun').text("Imagen_Modificada con Operador Thresholding");
                break;
            case "2":
                $('#bot').children('#const1').show();
                $('#bot').children('#const2').show();
                $('#bot').children('#const3').hide();
                $('#bot').children('#const4').hide();
                $('#constante').val(0);
                $('#constante1').val(100);
                $('#resul').children('#enun').text("Imagen_Modificada con Operador Contrast stretching");
                break;
            case "3":
                $('#bot').children('#const1').show();
                $('#bot').children('#const2').show();
                $('#bot').children('#const3').show();
                $('#bot').children('#const4').show();
                $('#constante').val(0);
                $('#constante1').val(0);
                $('#constante2').val(0);
                $('#constante3').val(0);
                $('#resul').children('#enun').text("Imagen_Modificada con Operador Ecualización de Histograma");
                break;
            case "4":
                $('#bot').children('#const1').show();
                $('#bot').children('#const2').hide();
                $('#bot').children('#const3').hide();
                $('#bot').children('#const4').hide();
                $('#resul').children('#enun').text("Imagen_Modificada con Operador Logaritmico");
                break;
            case "5":
                $('#bot').children('#const1').show();
                $('#bot').children('#const2').hide();
                $('#bot').children('#const3').hide();
                $('#bot').children('#const4').hide();
                $('#resul').children('#enun').text("Imagen_Modificada con Operador Raiz");
                break;
            case "6":
                $('#bot').children('#const1').show();
                $('#bot').children('#const2').show();
                $('#bot').children('#const3').hide();
                $('#bot').children('#const4').hide();
                $('#resul').children('#enun').text("Imagen_Modificada con Operador Exponencial");
                break;
            case "7":
                $('#bot').children('#const1').show();
                $('#bot').children('#const2').show();
                $('#bot').children('#const3').hide();
                $('#bot').children('#const4').hide();
                $('#resul').children('#enun').text("Imagen_Modificada con Operador Raise to Power");
                break;
             default:
                console.log('default');
        }

    });

});