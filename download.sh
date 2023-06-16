case $1 in
    tears)
        echo "Downloading tears of steel"
        wget http://ftp.nluug.nl/pub/graphics/blender/demo/movies/ToS/tears_of_steel_1080p.mov
        ;;

    golden)
        echo "Downloading golden eye"
        yt-dlp https://www.youtube.com/watch?v=OMgIPnCnlbQ
        ;;
    *)
        echo "Unkown resource: $1"
        echo "try: $0 tears|golden"
        ;;
esac

