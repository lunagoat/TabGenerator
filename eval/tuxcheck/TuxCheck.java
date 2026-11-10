// Opens a .gp5 file with TuxGuitar's own Guitar Pro reader and prints what
// it sees, to check that a generated file loads there. See eval/tuxcheck.sh.
import app.tuxguitar.io.base.TGSongReaderHandle;
import app.tuxguitar.io.gtp.GP5InputStream;
import app.tuxguitar.io.gtp.GTPSettings;
import app.tuxguitar.song.factory.TGFactory;
import app.tuxguitar.song.models.*;
import java.io.*;
import java.util.Iterator;

public class TuxCheck {
    public static void main(String[] args) throws Exception {
        for (String path : args) {
            TGSongReaderHandle handle = new TGSongReaderHandle();
            handle.setFactory(new TGFactory());
            handle.setInputStream(new BufferedInputStream(new FileInputStream(path)));
            new GP5InputStream(new GTPSettings()).read(handle);
            TGSong song = handle.getSong();
            System.out.println(new File(path).getName() + ": " + song.countTracks() + " tracks, " + song.countMeasureHeaders() + " bars");
            Iterator<TGTrack> tracks = song.getTracks();
            while (tracks.hasNext()) {
                TGTrack track = tracks.next();
                int notes = 0, maxFret = 0, badString = 0;
                Iterator<TGMeasure> measures = track.getMeasures();
                while (measures.hasNext()) {
                    TGMeasure m = measures.next();
                    for (int b = 0; b < m.countBeats(); b++) {
                        TGVoice v = m.getBeat(b).getVoice(0);
                        for (int n = 0; n < v.countNotes(); n++) {
                            TGNote note = v.getNote(n);
                            notes++;
                            maxFret = Math.max(maxFret, note.getValue());
                            if (note.getString() < 1 || note.getString() > track.stringCount()) badString++;
                        }
                    }
                }
                StringBuilder tuning = new StringBuilder();
                for (int s = 1; s <= track.stringCount(); s++) tuning.append(track.getString(s).getValue()).append(' ');
                System.out.println("  " + track.getName() + ": " + track.stringCount() + " strings [" + tuning.toString().trim()
                        + "], " + notes + " notes, highest fret " + maxFret + (badString > 0 ? ", " + badString + " NOTES ON MISSING STRINGS" : ""));
            }
        }
    }
}
